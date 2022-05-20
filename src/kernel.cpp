#include "kernel.h"

namespace cosyr { namespace kernel {

/* -------------------------------------------------------------------------- */
KOKKOS_FUNCTION
void emit_wavefronts(Kokkos::View<double*> emit_info,
                     Kokkos::View<double*> emit_current,
                     Kokkos::View<int*> num_active,
                     Kokkos::View<double*[DIM]> loaded_wavelet,
                     Kokkos::vector<Kokkos::View<double*>> loaded_field,
                     Kokkos::View<double*[DIM]> emitted_wavelet,
                     Kokkos::vector<Kokkos::View<double*>> emitted_field,
                     int num_loaded_wavelets,
                     int num_wavefronts,
                     int num_dirs,
                     int step,
                     int j,
                     double cos_mc_angle,
                     double sin_mc_angle,
                     double mesh_radius,
                     bool use_subcycle_wavelets) {

  // handle loaded wavelets first
  num_active[j] = (use_subcycle_wavelets ? num_loaded_wavelets : 0);

  // only shift non-reference (i.e. j>0) particle's loaded wavelets
  if (j > 0 and use_subcycle_wavelets) {
    int const jq = j * NUM_EMT_QUANTITIES;
    // projection of particle location to mesh coordinate (x',y')
    double const emit_x = emit_current[jq + EMT_POS_X];
    double const emit_y = emit_current[jq + EMT_POS_Y];
    double const emit_x_local = emit_x * cos_mc_angle - emit_y * sin_mc_angle;
    double const emit_y_local = emit_x * sin_mc_angle + emit_y * cos_mc_angle - mesh_radius;

    // loop over loaded wavelets and add shift
    for (int index_old = 0; index_old < num_loaded_wavelets; index_old++) {
      // add shift for wavelet ilw of particle j from wavelet ilw of particle 0 (reference particle)
      int const index_new = index_old + j * num_loaded_wavelets;
      loaded_wavelet(index_new, WT_POS_X) = loaded_wavelet(index_old, WT_POS_X) + emit_x_local;
      loaded_wavelet(index_new, WT_POS_Y) = loaded_wavelet(index_old, WT_POS_Y) + emit_y_local;
    }
  }

  // newest (generated at current step) wavefront, handle separately
  // store position of newest wavefront
  int index_wavefront = j * num_wavefronts;
  int index_info_wave = (index_wavefront + step) * NUM_EMT_QUANTITIES; // wavefront index in array of emission info.
  int index_wavelet = index_wavefront * num_dirs;

  double const cx = emit_info(index_info_wave + EMT_POS_X) * cos_mc_angle
                  - emit_info(index_info_wave + EMT_POS_Y) * sin_mc_angle;
  double const cy = emit_info(index_info_wave + EMT_POS_X) * sin_mc_angle
                  + emit_info(index_info_wave + EMT_POS_Y) * cos_mc_angle - mesh_radius;

  // TODO: check emitted wavelet coords here
  for (int k = 0; k < num_dirs; k++, index_wavelet++) { // loop over field directions
    emitted_wavelet(index_wavelet, WT_POS_X) = cx;
    emitted_wavelet(index_wavelet, WT_POS_Y) = cy;
    for (int d = 0; d <= DIM; ++d) {
      emitted_field[d](index_wavelet) = 0.0;
    }
  }

  num_active[j] += num_dirs;
}

/* -------------------------------------------------------------------------- */
KOKKOS_FUNCTION
int intersect_wavefronts_mesh(Kokkos::View<double*> emit_info,
                              int index_emit_wave,
                              double dt_emit,
                              double mesh_halfwidth_x,
                              double mesh_halfwidth_y,
                              double cos_mc_angle,
                              double sin_mc_angle,
                              double mesh_radius,
                              double angle_range[4][2],
                              double* angle_running_sum) {

  //double psi = dt_emt * beta;    // retarded angle
  // projection of emission location to mesh coordinate (x',y')
  double const cx = emit_info(index_emit_wave + EMT_POS_X) * cos_mc_angle
                  - emit_info(index_emit_wave + EMT_POS_Y) * sin_mc_angle;
  double const cy = emit_info(index_emit_wave + EMT_POS_X) * sin_mc_angle
                  + emit_info(index_emit_wave + EMT_POS_Y) * cos_mc_angle - mesh_radius;

  // find intersection of wavefront and mesh edges
  double const dt_emit2   = dt_emit * dt_emit;
  double const dis_xleft  = fabs(cx + mesh_halfwidth_x);
  double const dis_xright = mesh_halfwidth_x - cx;
  double const dis_ytop   = mesh_halfwidth_y - cy;
  double const dis_ybottom = -mesh_halfwidth_y - cy;
  bool wavefront_inside_mesh = true;

  // angles for intersections for top, right, bottom and left edges.
  // range (-PI, PI), relative to mesh x' axis.
  double intersection_angle_list[8] = {-10, -10,
                                       -10, -10,
                                       -10, -10,
                                       -10, -10};
  int iang[8] = {0, 0,
                 0, 0,
                 0, 0,
                 0, 0}; // flags for intersection, two per edge

  // check wavefront mesh boundary intersection
  if (fabs(dis_ytop) <= dt_emit) { // intersect top boundary
    wavefront_inside_mesh = false;
    double x1 = sqrt(dt_emit2 - dis_ytop * dis_ytop);
    double x2 = cx + x1;
    x1 = cx - x1;
    if (x1 > -mesh_halfwidth_x && x1 < mesh_halfwidth_x) { // intersections at corners are excluded
      double ang_top1 = acos(dis_ytop / dt_emit) + M_PI_2;
      intersection_angle_list[0] = ang_top1; // angle for left intersection
      iang[0] += 1;
    }
    if (x2 > -mesh_halfwidth_x && x2 < mesh_halfwidth_x) {
      double ang_top2 = asin(dis_ytop / dt_emit);
      intersection_angle_list[1] = ang_top2; // angle for right intersection
      iang[1] += 1;
    }
  }
  if (fabs(dis_xright) <= dt_emit) { // intersect right boundary
    wavefront_inside_mesh = false;
    double y1 = sqrt(dt_emit2 - dis_xright * dis_xright);
    double y2 = cy - y1;
    y1 = cy + y1;
    if (y1 > -mesh_halfwidth_y && y1 < mesh_halfwidth_y) {
      double ang_right1 = acos(dis_xright / dt_emit);
      intersection_angle_list[2] = ang_right1; // angle for top intersection
      iang[2] += 1;
    }
    if (y2 > -mesh_halfwidth_y && y2 < mesh_halfwidth_y) {
      double ang_right2 = -acos(dis_xright / dt_emit);
      intersection_angle_list[3] = ang_right2; // angle for bottom intersection
      iang[3] += 1;
    }
  }
  if (fabs(dis_ybottom) <= dt_emit) { // intersect bottom boundary
    wavefront_inside_mesh = false;
    double x1 = sqrt(dt_emit2 - dis_ybottom * dis_ybottom);
    double x2 = cx - x1;
    x1 = cx + x1;
    if (x1 > -mesh_halfwidth_x && x1 < mesh_halfwidth_x) {
      double ang_bottom1 = asin(dis_ybottom / dt_emit);
      intersection_angle_list[4] = ang_bottom1; // angle for right intersection
      iang[4] += 1;
    }
    if (x2 > -mesh_halfwidth_x && x2 < mesh_halfwidth_x) {
      double ang_bottom2 = acos(dis_ybottom / dt_emit) + M_PI_2;
      if (dis_ybottom < 0.) { ang_bottom2 -= 2*M_PI; }
      intersection_angle_list[5] = ang_bottom2; // angle for left intersection
      iang[5] += 1;
    }
  }
  if (fabs(dis_xleft) <= dt_emit) { // intersect left boundary
    wavefront_inside_mesh = false;
    double y1 = sqrt(dt_emit2 - dis_xleft * dis_xleft);
    double y2 = cy + y1;
    y1 = cy - y1;
    if (y1 > -mesh_halfwidth_y && y1 < mesh_halfwidth_y) {
      double ang_left1 = - acos(dis_xleft / dt_emit);
      intersection_angle_list[6] = ang_left1; // angle for top intersection
      iang[6] += 1;
    }
    if (y2 > -mesh_halfwidth_y && y2 < mesh_halfwidth_y) {
      double ang_left2 = acos(dis_xleft / dt_emit);
      intersection_angle_list[7] = ang_left2; // angle for bottom intersection
      iang[7] += 1;
    }
  }

  int range_count = 0;

  if (!wavefront_inside_mesh) {
    int ias, iae;
    for (ias = 1; ias<8; ) {
      if (iang[ias]==1) {
        // found angle range start
        angle_range[range_count][0] = intersection_angle_list[ias];
        // check following even intersection points for range end
        for (iae = ias+1; iae<(ias+9); iae +=2) {
          int iaem = iae % 8;
          if (iang[iaem] == 1) {
            // found angle range end
            //make sure ending angle < starting angle
            if (intersection_angle_list[iaem] > angle_range[range_count][0]) {
              angle_range[range_count][1] = intersection_angle_list[iaem] - 2 * M_PI;
            } else {
              angle_range[range_count][1] = intersection_angle_list[iaem];
            }
            angle_running_sum[range_count+1] = angle_running_sum[range_count]
                                             + (angle_range[range_count][0] - angle_range[range_count][1]);
            range_count +=1;
            break;
          }
        }
        if (iae > ias) {
          ias = iae + 1; // go to next odd intersection point following last ending intersection
        } else {
          ias +=2; // go to next odd intersection point
        }
      } else {
        // go to next odd intersection point
        ias +=2;
      }
    }
  } else {
    // wavefront is completely inside mesh, then use default full range (PI, -PI)
    angle_running_sum[range_count+1] = angle_range[range_count][0] - angle_range[range_count][1];
    range_count += 1;
  }

  return range_count;
}

/* -------------------------------------------------------------------------- */
KOKKOS_FUNCTION
void calculate_fields(Kokkos::View<double*> emit_info,
                      Kokkos::View<int*> num_active,
                      Kokkos::View<double*[DIM]> emitted_wavelet,
                      Kokkos::vector<Kokkos::View<double*>> emitted_field,
                      int index_particle,
                      int index_wavelet,
                      int index_emit_wave,
                      double dt_emt,
                      double q,
                      double gamma_ref,
                      double min_emit_angle,
                      int num_dirs,
                      int range_count,
                      double angle_range[4][2],
                      const double* angle_running_sum,
                      double mesh_center_x,
                      double mesh_center_y,
                      double mesh_center_angle,
                      double cos_mc_angle,
                      double sin_mc_angle) {

  double const r2          = std::pow(dt_emt, 2.0);
  double const betaprime_x = emit_info(index_emit_wave + EMT_VEL_X);
  double const betaprime_y = emit_info(index_emit_wave + EMT_VEL_Y);
  double const gamma_prime = emit_info(index_emit_wave + EMT_GAMMA);
  double const gamma2      = std::pow(gamma_ref, 2.0); 
  double const g2_prime    = std::pow(gamma_prime, 2.0); 
  double beta = sqrt(1.0 - 1.0/gamma2);
  double const syn_cone    = min_emit_angle / gamma_prime;
  double syn_cone_ulim = std::atan2(betaprime_y, betaprime_x);
  double const syn_cone_llim = syn_cone_ulim - syn_cone;
  syn_cone_ulim += syn_cone;
  double const qgr2 = q / g2_prime / r2; // this is q/(gamma^2*r^2) in velocity field
  double const qr   = q / dt_emt;        // this is q/r in acceleration field

  int ndirs_local = num_dirs;
  int ndirs_range[5] = {0, 0, 0, 0, 0};

  if (range_count==0) {
    // wavefront completely outside mesh
    ndirs_local = 0;
  } else {
    // decide number of wavelets in each range
    for (int irange = 0; irange <= range_count; irange++) {
      ndirs_range[irange] = static_cast<int>(std::round(ndirs_local
                                             * angle_running_sum[irange]
                                             / angle_running_sum[range_count]));
    }
  }

  // TODO: should include some margin for interpolation at mesh edge
  double emt_angle_interval = angle_running_sum[range_count]/(num_dirs-1);
  double emt_angle;

  int current_range = 0;
  num_active[index_particle] += ndirs_local;

  for (int k = 0; k < ndirs_local; k++, index_wavelet++) { // loop over wavelet directions
    // update the range start angle if wavelet is in a new range
    if (k == ndirs_range[current_range]) {
      emt_angle = angle_range[current_range][0] - mesh_center_angle;
      current_range += 1;
    }

    // wavelet directions in global coordinate
    double const n_x = cos(emt_angle);
    double const n_y = sin(emt_angle);

    // calculate global (x,y) coordinate of wavelet
    double wpx = emit_info(index_emit_wave + EMT_POS_X) + n_x * dt_emt;
    double wpy = emit_info(index_emit_wave + EMT_POS_Y) + n_y * dt_emt;

    #ifdef PROJECT_WITH_OWN_ORIGIN
      // calculate wavelet offset to particle's own origin
      wpy -= emit_info(index_emit_wave + EMT_POS_Y) - mesh_radius;
    #endif
    double const dis = sqrt(wpx * wpx + wpy * wpy);

    // convert to local (x',y') coordinate
    double const psi = dt_emt * beta;
    double const wpxm = wpx - mesh_center_x;
    double const wpym = wpy - mesh_center_y;
    double const wpx_prime = wpxm * cos_mc_angle - wpym * sin_mc_angle;
    double const wpy_prime = wpxm * sin_mc_angle + wpym * cos_mc_angle;

    emitted_wavelet(index_wavelet, WT_POS_X) = wpx_prime;
    emitted_wavelet(index_wavelet, WT_POS_Y) = wpy_prime;

    // (wpx, wpy) now is the vector pointing to the wavelet in global coordinate
    // for field projection purpose.
    wpx /= dis;
    wpy /= dis;
    double const accel_x = emit_info(index_emit_wave + EMT_ACC_X);
    double const accel_y = emit_info(index_emit_wave + EMT_ACC_Y);

    // note the field calculation is done in global coordinate
    if (emt_angle < syn_cone_ulim and emt_angle > syn_cone_llim) {
      // TODO: better way to handle emission within the synchrotron cone
      for (int d = 0; d <= DIM; ++d) {
        emitted_field[d](index_wavelet) = 0.0;
      }
    } else {
      double const one_minus_n_dot_betaprime = 1.0 - n_x * betaprime_x - n_y * betaprime_y;
      double const inv_denom = pow(one_minus_n_dot_betaprime, -3.0);
      double const qr_inv_denom = qr * inv_denom;

      #ifdef UPDATE_VELOCITY_FIELD
        // computes and updates the velocity field
        double const qgr2_inv_denom = qgr2 * inv_denom;
        double vel_fld_x = qgr2_inv_denom * (n_x - betaprime_x);
        double vel_fld_y = qgr2_inv_denom * (n_y - betaprime_y);
      #endif

      // computes and updates the acceleration/radiation field
      double const acc_fld_x = qr_inv_denom * ((n_x * accel_x + n_y * accel_y)
                               * (n_x - betaprime_x) - one_minus_n_dot_betaprime * accel_x);
      double const acc_fld_y = qr_inv_denom * ((n_x * accel_x + n_y * accel_y)
                               * (n_y - betaprime_y) - one_minus_n_dot_betaprime * accel_y);

      double Brad_z = -acc_fld_x * n_y + acc_fld_y * n_x; //B^{rad}_{z}
      double Erad_t =  acc_fld_x * wpx + acc_fld_y * wpy; // E^{rad}_{t}
      //double Erad_t =  acc_fld_x * sin_mc_angle + acc_fld_y * cos_mc_angle; // E^{rad}_{t}

      // beta is along direction of the vector (cos_mc_angle, -sin_mc_angle)
      double betaprime_dot_beta = beta*(cos_mc_angle * betaprime_x - sin_mc_angle * betaprime_y);

      #ifdef MIX_KERNEL
      emitted_field[0](index_wavelet) = Brad_z; 
      //emitted_field[0](index_wavelet) = qr / one_minus_n_dot_betaprime; // phi  
      emitted_field[1](index_wavelet) = Erad_t - beta*Brad_z; 
      //emitted_field[2](index_wavelet) = qr * (1.0-betaprime_dot_beta) / one_minus_n_dot_betaprime; // phi-beta*As 
      emitted_field[2](index_wavelet) = qr * (1.0-beta*beta*cos(psi)) / one_minus_n_dot_betaprime; // phi-beta*As 

      #else 
      emitted_field[0](index_wavelet) = Brad_z; //B^{rad}_{z}
      emitted_field[1](index_wavelet) = Erad_t; // E^{rad}_{t}
      emitted_field[2](index_wavelet) =  acc_fld_x * wpy - acc_fld_y * wpx; // E^{rad}_{s}
      #endif
    }

    emt_angle -= emt_angle_interval;
  } // end of k
}

/* -------------------------------------------------------------------------- */
}} // namespace 'cosyr::kernel'


