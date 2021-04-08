#include <iostream>
#include <cmath>
#include <functional>

#include "beam.h"

namespace cosyr {

/* -------------------------------------------------------------------------- */
Beam::Beam(Input const& in_input) : input(in_input) {

  int const num_particles = input.kernel.num_particles;
  int const num_steps = input.kernel.num_step;
  int const num_wavefronts = input.kernel.num_wavefronts;

  q = std::copysign(1.0, input.kernel.qm)
    * input.beam.charge * NUM_ELECTRON_nC
    / num_particles / input.mpi.num_ranks;

  qm = input.kernel.qm;

  particles.resize(num_particles);
  Kokkos::resize(trajectory, num_particles * num_steps * NUM_TRAJ_QUANTITIES);
  Kokkos::resize(emit_info, num_particles * num_wavefronts * NUM_EMT_QUANTITIES);
  Kokkos::resize(emit_current, num_particles * NUM_EMT_QUANTITIES);
  host_emit_current =  Kokkos::create_mirror_view(emit_current);

  if (input.beam.found) {
    copy(num_particles, input.beam.particles);
  } else {
    init(input.kernel.pos_elec, input.kernel.mom_elec);
  }

  if (input.mpi.rank == 0)
    std::cout << "Beam created, charge per particle = " << q << " (e)." << std::endl;
}

/* -------------------------------------------------------------------------- */
void Beam::move_reference(int index_particle,
                          Mesh const& mesh,
                          Traject traject,
                          const double* motion_params,
                          int step,
                          double t,
                          double dt) {

  // advance position for 1st half step: x_{0} -> x_{1/2}
  // reference particle has global coordinate but relative to it is initial position
  auto position = Cabana::slice<Position>(particles);
  auto momentum = Cabana::slice<Momentum>(particles);
  auto lorentz  = Cabana::slice<Gamma>(particles);

  double gamma = lorentz(index_particle);
  double const hdt = 0.5 * dt;
  double momenta[DIM] = {0.0};
  double const mesh_radius = 1.0; //TODO: pass this from caller
  double const vx = momentum(index_particle, PART_MOM_X) / gamma;
  double const vy = momentum(index_particle, PART_MOM_Y) / gamma;

  double new_x = position(index_particle, PART_POS_X) + vx * hdt; // time center position
  double new_y = position(index_particle, PART_POS_Y) + vy * hdt;
  reference.coords[0] = new_x;
  reference.coords[1] = new_y;

#if DIM==3
  double vz = momentum(index_particle, PART_MOM_Z) / gamma;
  double new_z = position(index_particle, PART_POS_Z) + vz * hdt;
  reference.coords[2] = new_z;
#endif

  // velocity update: v_{0} -> v_{1}
  calculate_moment(t, new_x, traject, motion_params, momenta, &gamma);
  momentum(index_particle, PART_MOM_X) = momenta[0];
  momentum(index_particle, PART_MOM_Y) = momenta[1];
  reference.moment[0] = momenta[0];
  reference.moment[1] = momenta[1];

  lorentz(index_particle) = gamma;
  double new_vx = momenta[0] / gamma;
  double new_vy = momenta[1] / gamma;

#if DIM==3
  momentum(index_particle, PART_MOM_Z) = momenta[2];
  reference.moment[2] = momenta[2];
  double new_vz = momenta[2] / gamma;
#endif

  // only record emission info at t_{1/2} here,
  // wavefronts and fields are now calculated on device before interpolation
  host_emit_current[EMT_TIME]  = t;
  host_emit_current[EMT_POS_X] = new_x;
  host_emit_current[EMT_POS_Y] = new_y;
  // note we only know acceleration at half step
  host_emit_current[EMT_ACC_X] = (new_vx - vx) / dt;
  host_emit_current[EMT_ACC_Y] = (new_vy - vy) / dt;
  host_emit_current[EMT_VEL_X] = 0.5 * (new_vx + vx);
  host_emit_current[EMT_VEL_Y] = 0.5 * (new_vy + vy);
  host_emit_current[EMT_GAMMA] = gamma;

#if DIM==3
  host_emit_current[EMT_POS_Z] = new_z;
  host_emit_current[EMT_ACC_Z] = (new_vz - vz) / dt;
  host_emit_current[EMT_VEL_Z] = 0.5 * (new_vz + vz);
#endif

  // advance position for 2nd half step: x_{1/2} -> x_{1}
  new_x += new_vx * hdt;
  new_y += new_vy * hdt;

  // enforce reference particle on circular trajectory
  if (traject == CIRCULAR) {
    new_y = std::sqrt(std::pow(mesh_radius, 2) - std::pow(new_x, 2));
  }

  position(index_particle, PART_POS_X) = new_x;
  position(index_particle, PART_POS_Y) = new_y;

  int const index_traject = step * NUM_TRAJ_QUANTITIES;
  trajectory[index_traject + TRAJ_POS_X] = new_x;
  trajectory[index_traject + TRAJ_POS_Y] = new_y;

#if DIM==3
  new_z += new_vz * hdt;
  position(index_particle, PART_POS_Z) = new_z;
  trajectory(index_traject + TRAJ_POS_Z) = new_z;
#endif
}

/* -------------------------------------------------------------------------- */
void Beam::move_others(Mesh const& mesh,
                       const double* motion_params,
                       int step,
                       double dt,
                       double time) {

  double half_dt = dt/2.0;
  double qmhdt = qm * half_dt;

  auto position = Cabana::slice<Position>(particles);
  auto momentum = Cabana::slice<Momentum>(particles);
  auto gammas   = Cabana::slice<Gamma>(particles);

  auto b_trans = Cabana::slice<F1>(mesh.fields);
  auto e_trans = Cabana::slice<F2>(mesh.fields);
  auto e_long  = Cabana::slice<F3>(mesh.fields);

  if (not mesh.is_updated) {
    throw std::runtime_error("mesh should be updated first");
  }

  double mesh_center_x = mesh.center.position[0];
  double mesh_center_y = mesh.center.position[1];
  double mesh_center_angle = std::atan2(mesh_center_x, mesh_center_y);
  double sin_mc_angle = std::sin(mesh_center_angle);
  double cos_mc_angle = std::cos(mesh_center_angle);
  double mesh_dx = mesh.h[0];
  double mesh_dy = mesh.h[1];
  int nx = mesh.resolution[0];
  int ny = mesh.resolution[1];
  int half_nx = (nx - 1) / 2;
  int half_ny = (ny - 1) / 2;

  // for circular trajectory, params[1] should be radius in cm.
  const double unit_fac = std::pow(ELECTRON_CHARGE, 2)
                          / (motion_params[1] * ELECTRON_MASS
                            * std::pow(LIGHT_SPEED, 2));

  double gamma0 = gammas(0);

  const double one = 1.;
  double bz0 = -1.0 * std::sqrt(gamma0 * gamma0 - 1.0);

#ifdef DEBUG
  std::cout << std::setprecision(12);
  std::cout << "mesh_center = " << mesh_center_x << ", " << mesh_center_y << std::endl;
  std::cout << "unit_fac, q = " << unit_fac << ", " << q << std::endl;
  std::cout << "particle 1: "
            << position.access(0, 1, PART_POS_X) <<", "
            << position.access(0, 1, PART_POS_Y) <<", "
            << momentum.access(0, 1, PART_MOM_X) << ", "
            << momentum.access(0, 1, PART_MOM_Y) << std::endl;
#endif

  auto _push = KOKKOS_LAMBDA(int s, int a) {

    int j = 16 * s + a;
    // index of trajectory step i for particle j
    int ips = step + j * input.kernel.num_step;
    int ipq = j * NUM_EMT_QUANTITIES;
    int ipsq = ips * NUM_TRAJ_QUANTITIES;
    int jq = j * NUM_PARTICLE_QUANTITIES;
    double new_x, new_y, ux, uy, vx, vy;
    double uz = 0.0, vz = 0.0;

    // advance position for 1st half step: x_{0} -> x_{1/2}
    ux = momentum.access(s, a, PART_MOM_X);
    uy = momentum.access(s, a, PART_MOM_Y);
    double inv_gamma = one / gammas.access(s, a);
    vx = ux * inv_gamma;
    vy = uy * inv_gamma;
    new_x = position.access(s, a, PART_POS_X) + vx * half_dt; // time center position
    new_y = position.access(s, a, PART_POS_Y) + vy * half_dt;
    #if DIM == 3
      uz = momentum.access(s, a, PART_MOM_Z);
      vz = uz * inv_gamma;
      double new_z = position.access(s, a, PART_POS_Z) + vz * half_dt;
    #endif

    // project into local coordinates
    double x_local = new_x - mesh_center_x;
    double y_local = new_y - mesh_center_y;
    double cx = x_local * cos_mc_angle - y_local * sin_mc_angle;
    double cy = x_local * sin_mc_angle + y_local * cos_mc_angle;
    #ifdef DEBUG
      std::cout << std::setprecision(12);
      std::cout << "part " << j << " (s=" << s << " , a=" << a << ") of " << particles.size()
                << ": new_x, new_y, x_local, y_local, cx, cy = " << new_x << ", "  << new_y
                << ", " << x_local << ", " << y_local << ", " << cx << ", " << cy << std::endl;
    #endif
    cx = cx / mesh_dx;
    cy = cy / mesh_dy;
    int ix = int(floor(cx));
    int iy = int(floor(cy));
    cx = cx - ix;
    cy = cy - iy;
    ix += half_nx;
    iy += half_ny;
    #ifdef DEBUG
      std::cout << std::setprecision(12);
      std::cout << "mesh_dx, mesh_dy, half_nx, half_ny, ix, iy, cx, cy = "
                << mesh_dx << ", " << mesh_dy << ", " << half_nx << ", " << half_ny
                << ", " << ix << ", " << iy << ", " << cx << ", " << cy << std::endl;
    #endif

    // interpolate fields
    int top_left = ix * ny + iy + 1;
    int top_right = (ix+1) * ny + iy + 1;
    int bottom_left = ix * ny + iy;
    int bottom_right = (ix+1) * ny + iy;
    double cx1 = one - cx, cy1 = one - cy;
    double el = cx * cy * e_long(top_right) + cx * cy1 * e_long(bottom_right)
                + cx1 * cy * e_long(top_left) + cx1 * cy1 * e_long(bottom_left);
    double et = cx * cy * e_trans(top_right) + cx * cy1 * e_trans(bottom_right)
                + cx1 * cy * e_trans(top_left) + cx1 * cy1 * e_trans(bottom_left);
    double bz = cx * cy * b_trans(top_right) + cx * cy1 * b_trans(bottom_right)
                + cx1 * cy * b_trans(top_left) + cx1 * cy1 * b_trans(bottom_left);

    el *= unit_fac;
    et *= unit_fac;
    bz *= unit_fac;
    #ifdef DEBUG
      std::cout << std::setprecision(12);
      std::cout << "top_right, bottom_right, top_left, bottom_left, size = "
                << top_right << ", " << bottom_right << ", " << top_left << ", "
                << bottom_left << ", " << mesh.num_points << std::endl;
      std::cout << "e_long = " << e_long(top_right) << ", " << e_long(bottom_right) << ", "
                << e_long(top_left) << ", " << e_long(bottom_left) << std::endl;
      std::cout << "e_trans = " << e_trans(top_right) << ", " << e_trans(bottom_right) << ", "
                << e_trans(top_left) << ", " << e_trans(bottom_left) << std::endl;
      std::cout << "b_trans = " << b_trans(top_right) << ", " << b_trans(bottom_right) << ", "
                << b_trans(top_left) << ", " << b_trans(bottom_left) << std::endl;
      std::cout << "el, et, bz = " << el << ", " << et << ", " << bz << std::endl;
    #endif

    double ex = el * cos_mc_angle + et * sin_mc_angle;
    double ey = - el * sin_mc_angle + et * cos_mc_angle;
    double hax = qmhdt * ex;
    double hay = qmhdt * ey;
    double haz = qmhdt * 0.0;
    double bx = 0.0;
    double by = 0.0;
    bz += bz0;

    // momentum update (Boris): u_{0} -> u_{1}
    // first electric half push
    ux += hax;
    uy += hay;
    double u2 = ux * ux + uy * uy;
    #if DIM == 3
      uz += haz;
      u2 += uz * uz;
    #endif
    inv_gamma = one / std::sqrt(u2 + one);

    // rotation
    double v0, v1, v2, v3, v4;
    v0  = qmhdt * inv_gamma;
    v1  = bx*bx + by*by + bz*bz;    // b^2
    #ifdef BORIS_CORRECTION
      v2 = (v0*v0)*v1;
      v3 = v0*(one+v2*(one_third+v2*two_fifteenths));  // t
      v4 = v3/(one+v1*(v3*v3));      // s
    #else
      // basic Boris
      v3  = v0;                       // t
      v4  = v3/(one+v1*(v3*v3));      // s
    #endif
    v4 += v4;                       // x 2
    v0  = ux + v3*( uy*bz - uz*by );       // u_prime = u_- + u_- x t
    v1  = uy + v3*( uz*bx - ux*bz );
    v2  = uz + v3*( ux*by - uy*bx );
    ux += v4*( v1*bz - v2*by );            // u_+ = u_- + u_prime x s
    uy += v4*( v2*bx - v0*bz );
    #if DIM == 3
      uz += v4*( v0*by - v1*bx );
    #endif

    // second electric half push
    ux += hax;
    uy += hay;
    u2 = ux * ux + uy * uy;
    #if DIM==3
      uz += haz;
      u2 += uz * uz;
    #endif

    double gamma = std::sqrt(u2 + one);
    double hdt_inv_gamma = half_dt / gamma;
    double new_vx = ux / gamma;
    double new_vy = uy / gamma;

    // only record emission info at t_{1/2} here,
    // wavefronts and fields are now calculated on device before interpolation.
    host_emit_current[ipq + EMT_TIME]  = time;
    host_emit_current[ipq + EMT_POS_X] = new_x;
    host_emit_current[ipq + EMT_POS_Y] = new_y;
    // note we only know acceleration at half step
    host_emit_current[ipq + EMT_ACC_X] = (new_vx - vx) / dt;
    host_emit_current[ipq + EMT_ACC_Y] = (new_vy - vy) / dt;
    host_emit_current[ipq + EMT_VEL_X] = 0.5 * (new_vx + vx);
    host_emit_current[ipq + EMT_VEL_Y] = 0.5 * (new_vy + vy);
    host_emit_current[ipq + EMT_GAMMA] = gamma;
    #if DIM==3
      double new_vz = uz / gamma;
      mirror_emit_current[ipq + EMT_POS_Z] = new_z;
      mirror_emit_current[ipq + EMT_ACC_Z] = (new_vz - vz) / dt;
      mirror_emit_current[ipq + EMT_VEL_Z] = 0.5 * (new_vz + vz);
    #endif

    // store velocity and advance position for 2nd half step: x_{1/2} -> x_{1}
    new_x += ux * hdt_inv_gamma; // time center position
    new_y += uy * hdt_inv_gamma;
    position.access(s, a, PART_POS_X) = new_x;
    position.access(s, a, PART_POS_Y) = new_y;
    trajectory[ipsq + TRAJ_POS_X] = new_x;
    trajectory[ipsq + TRAJ_POS_Y] = new_y;
    momentum.access(s, a, PART_MOM_X) = ux;
    momentum.access(s, a, PART_MOM_Y) = uy;
    gammas.access(s, a) = gamma;
    #if DIM==3
      new_z += uz * hdt_inv_gamma;
      position.access(s, a, PART_POS_Z) = new_z;
      momentum.access(s, a, PART_MOM_Z) = uz;
      trajectory[ipsq + TRAJ_POS_Z] = new_z;
    #endif
  }; // end particle loop

  Cabana::SimdPolicy<16, Host> simd_policy(1, particles.size() );
  Cabana::simd_parallel_for( simd_policy, _push, "push_particle" );
}

/* -------------------------------------------------------------------------- */
void Beam::print(int step, int index_particle) const {

  if (input.mpi.rank == 0) {
    int const i = index_particle * input.kernel.num_step + step;
    int iq = i * NUM_TRAJ_QUANTITIES;
    std::cout << "particle " << index_particle << ", position at time step "
              << step << ": (" << trajectory[iq + TRAJ_POS_X] << " , "
              << trajectory[iq + TRAJ_POS_Y] << ")" << std::endl;
  }
}

/* -------------------------------------------------------------------------- */
void Beam::init(const double *init_position, const double *init_momentum) const {

  if (input.mpi.rank == 0)
    std::cout << "Init particles to a given position and velocity ... ";

  // run on host
  auto range = HostRange(0, input.kernel.num_particles);
  auto position = Cabana::slice<Position>(particles);
  auto momentum = Cabana::slice<Momentum>(particles);

  Kokkos::parallel_for(range, [=](int j) {
    position(j, PART_POS_X) = init_position[0];
    position(j, PART_POS_Y) = init_position[1];
    momentum(j, PART_MOM_X) = init_momentum[0];
    momentum(j, PART_MOM_Y) = init_momentum[1];
#if DIM == 3
    position(j, PART_POS_Z) = init_position[2];
      momentum(j, PART_MOM_Z) = init_momentum[2];
#endif
  });

  if (input.mpi.rank == 0)
    std::cout << "done" << std::endl;
}

/* -------------------------------------------------------------------------- */
void Beam::copy(int count, const double* beam) {

  auto position = Cabana::slice<Position>(particles);
  auto momentum = Cabana::slice<Momentum>(particles);
  auto gamma    = Cabana::slice<Gamma>(particles);

#if DIM == 3
  double r0[DIM] = { beam[0], beam[1], beam[2] };
  double p0[DIM] = { beam[3], beam[4], beam[5] };
  double norm = std::sqrt(p0[0] * p0[0] + p0[1] * p0[1] + p0[2] * p0[2]);
  // unit vector along p0
  double p0_unit[DIM] = { p0[0] / norm, p0[1] / norm, p0[2] / norm };
  // unit vector perpendicular to p0 in bending plane
  double p0_perp[DIM] = { -p0_unit[1], p0_unit[2], p0_unit[3] };

#else
  double r0[DIM] = { 0.0, 0.0 };
  double p0[DIM] = { beam[2], beam[3] };
  double norm = std::sqrt(p0[0] * p0[0] + p0[1] * p0[1]);
  // unit vector along p0
  double p0_unit[DIM] = { p0[0] / norm, p0[1] / norm };
  // unit vector perpendicular to p0 in bending plane
  double p0_perp[DIM] = {-p0_unit[1], p0_unit[2] };
#endif

  // use global coordinates (relative to initial position of reference particle)
  // directly and copy into particle storage
  auto copy_beam = KOKKOS_LAMBDA(int index_particle) {

    int const index_coords = index_particle * NUM_PARTICLE_QUANTITIES;
    int const index_moment = index_coords + DIM;

    position(index_particle, PART_POS_X) = beam[index_coords];     // - r0[0];
    position(index_particle, PART_POS_Y) = beam[index_coords + 1]; // - r0[1];
    momentum(index_particle, PART_MOM_X) = beam[index_moment];
    momentum(index_particle, PART_MOM_Y) = beam[index_moment + 1];
    double psq = beam[index_moment] * beam[index_moment]
                 + beam[index_moment + 1] * beam[index_moment + 1];

#if DIM==3
    position(j, PART_POS_Z) = beam[index_coords + 2] - r0[2];
      momentum(j, PART_MOM_Z) = beam[index_moment + 2];
      psq += beam[index_moment + 2] * beam[index_moment + 2];
#endif

    gamma(index_particle) = std::sqrt(1.0 + psq);
  };

  // project global coordinates to local coordinates of reference particle
  // and copy into particle storage
  auto project_copy_beam = KOKKOS_LAMBDA(int index_particle) {

    int const index_coords = index_particle * NUM_PARTICLE_QUANTITIES;
    int const index_moment = index_coords + DIM;

    // project particle position and momentum
    // A//B = dot(A,B)/|B|, A \perp B = A - A//B
    double offset[DIM];
    double r_dot_p0 = 0.0;
    double p_dot_p0 = 0.0;
    double r_dot_p0perp = 0.0;
    double p_dot_p0perp = 0.0;
    double psq = 0.0;

    for (int i=0; i < DIM; i++) {
      offset[i] = beam[index_coords + i] - r0[i];
      double p_i = beam[index_moment + i];
      psq += p_i * p_i;
      r_dot_p0 += offset[i] * p0_unit[i];
      p_dot_p0 += p_i * p0_unit[i];
      r_dot_p0perp += offset[i] * p0_perp[i];
      p_dot_p0perp += p_i * p0_perp[i];
    }

    position(index_particle, PART_POS_X) = r_dot_p0;
    position(index_particle, PART_POS_Y) = r_dot_p0perp;
    momentum(index_particle, PART_MOM_X) = p_dot_p0;
    momentum(index_particle, PART_MOM_Y) = p_dot_p0perp;
    gamma(index_particle) = std::sqrt(1.0 + psq);

#if DIM==3
    position(j, PART_POS_Z) = offset[2];
      momentum(j, PART_MOM_Z) = beam[index_moment + 2];
#endif
  };

  // run on host
  Kokkos::parallel_for(HostRange(0, count), copy_beam, "copy_beam");
  //Kokkos::parallel_for(HostRange(0, count), project_copy_beam, "project_copy_beam");

#ifdef DEBUG
  std::cout << std::setprecision(12) << "initial particle 0: "
            << position(0, PART_POS_X) << ", "
            << position(0, PART_POS_Y) << ", "
            << momentum(0, PART_MOM_X) << ", "
            << momentum(0, PART_MOM_Y) << std::endl;

  std::cout << std::setprecision(12) << "initial particle 1: "
            << position(1, PART_POS_X) << ", "
            << position(1, PART_POS_Y) << ", "
            << momentum(1, PART_MOM_X) << ", "
            << momentum(1, PART_MOM_Y) << std::endl;
#endif
}

/* -------------------------------------------------------------------------- */
void Beam::calculate_moment(double time,
                            double x,
                            Traject traject,
                            const double* motion_params,
                            double* momenta,
                            double* gamma_e) const {

  *gamma_e = motion_params[0];
  double const beta = std::sqrt(1.0 - std::pow(*gamma_e, -2.0));
  double const p = std::sqrt(std::pow(*gamma_e, 2.0) - 1.0);

  switch (traject) {
    // straight line, params = [gamma_e, theta_0]
    case STRAIGHT: {
      double const theta_0 = motion_params[1];
      momenta[0] = p * std::cos(theta_0);
      momenta[1] = p * std::sin(theta_0);
    } break;

      // synchrotron, params = [gamma_e, radius]
    case CIRCULAR: {
      double const radius = motion_params[1];
      double const theta_0 = beta * time;

      if (time >= 0) {
        momenta[0] =  p * std::cos(theta_0);
        momenta[1] = -p * std::sin(theta_0);
#ifdef DEBUG
        double const rot_angle = std::asin(x / radius);
          std::cout << "angle dif = " << theta_0 - rot_angle << ", "
                    << theta_0 << ", " << time << ", " << rot_angle << ", "
                    << x << std::endl;
#endif
      } else {
        momenta[0] = p;
        momenta[1] = 0.0;
      }
    } break;

      // undulator, params = [gamma_e, omega, amp]
    case SINUSOIDAL: {
      auto const omega = motion_params[1];
      auto const amp   = motion_params[2];
      double const tan_theta = amp * omega * std::cos(omega * x);
      momenta[0] = p / std::sqrt(1.0 + std::pow(tan_theta, 2));
      momenta[1] = momenta[0] * tan_theta;
      //*theta_0 = atan(tan_theta);
    } break;

    default:
      throw std::runtime_error("invalid trajectory type");
  }

#if DIM==3
  momenta[2] = 0.0;
#endif
}

/* -------------------------------------------------------------------------- */
} // namespace 'cosyr'
