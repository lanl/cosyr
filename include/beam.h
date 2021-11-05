#pragma once

#include "cosyr.h"
#include "input.h"
#include "mesh.h"
#include "wavelet.h"

namespace cosyr {

/**
 * @brief Particle's information and function prototypes
 */
class Beam {
public:
  /**
   * @brief Data attributes of each particle.
   */
  enum Attributes { Position, Momentum, Gamma, EmitCoords };

  /**
   * @brief Particle information.
   *
   * - its position.
   * - its proper momentum (gamma * beta).
   * - its gamma at current step.
   */
  using Data = Cabana::MemberTypes<double[DIM], double[DIM], double, double[DIM]>;

  /**
   * @brief Create a particle beam.
   *
   * @param input: cached simulation parameters
   */
  explicit Beam(Input const& in_input);

  /**
   * @brief Push reference particle.
   *
   * @param index_particle: index of reference particle.
   * @param mesh: reference to moving mesh.
   * @param traject: trajectory type.
   * @param motion_params: electron motion parameters.
   * @param step: current step.
   * @param t: current time.
   * @param dt: time increment.
   */
  void move_reference(int index_particle,
                      Mesh const& mesh,
                      Traject traject,
                      const double* motion_params,
                      int step,
                      double t,
                      double dt);

  /**
   * @brief Push other particles.
   *
   * @param mesh: reference to moving mesh.
   * @param motion_params: electron motion parameters.
   * @param step: current step.
   * @param dt: time increment.
   * @param time: current time.
   */
  void move_others(Mesh const& mesh,
                   const double *motion_params,
                   int step,
                   double dt,
                   double time);

  /**
   * @brief Print information at given time step.
   *
   * @param step: current step
   * @param index_particle: index of particle
   */
  void print(int step, int index_particle) const;

  /**
   * @brief Create beam from a specified one.
   *
   * @param count: number of particles.
   * @param beam: coordinates and momenta of each particle.
   */
  void copy(int count, const double* beam);

  /**
   * @brief Charge per particle, in unit of positron charge.
   *
   */
  double q = -1.0;

  /**
   * @brief Charge over mass, in unit of e/m_e.
   *
   */
  double qm = -1.0;

  /**
   * @struct Reference particle.
   *
   */
  struct {
    /** its position */
    double coords[DIM] = {0};
    /** its moments */
    double momentum[DIM] = {0};
  } reference;

  /**
   * @brief Particles for current time step in moving mesh window, stored on host.
   *
   * Position[DIM]: in instantaneous local coordinate of the reference particle.
   * Momentum[DIM]: in instantaneous local coordinate of the reference particle.
   * Gamma: particle Lorentz factor
   */
  Cabana::AoSoA<Data, HostSpace, 16> particles;

  /**
   * @brief Beam trajectory history, only needed on the host.
   *
   */
  Kokkos::View<double*, HostSpace> trajectory;

  /**
   * @brief Wavefront emission related quantities.
   *
   * EMT_TIME: emission time of wavefront
   * EMT_POS_X: x coordinate of the origin of wavefront i at emission
   * EMT_POS_Y: y coordinate of the origin of wavefront i at emission
   * EMT_VEL_X: velocity history of electron emission in x direction
   * EMT_VEL_Y: velocity history of electron emission in y direction
   * EMT_ACC_X: acceleration history of electron emission in x direction
   * EMT_ACC_Y: acceleration history of electron emission in x direction
   */
  Kokkos::View<double*> emit_info;

  /**
   * @brief Current emitted wavefront related quantities.
   *
   */
  Kokkos::View<double*> emit_current;

  /**
   * @brief Host mirror for quantities related to current emitted wavefront.
   *
   */
  Kokkos::View<double*>::HostMirror host_emit_current;

private:
  /**
   * @brief Create beam with a given initial position and momentum.
   *
   * @param init_position: initial position.
   * @param init_momentum: initial momentum.
   */
  void init(const double *init_position, const double *init_momentum) const;

  /**
   * @brief Calculate proper momentum at a given position x.
   *
   * @param time: current time.
   * @param x: position of current particle.
   * @param traject: trajectory type.
   * @param motion_params: electron motion parameters
   * @param momenta: the computed momenta.
   * @param gamma_e: lorentz factor.
   */
  void calculate_momenta(double time,
                         double x,
                         Traject traject,
                         const double* motion_params,
                         double* momenta,
                         double* gamma_e) const;

  /**
   * @brief Reference to simulation parameters.
   *
   */
  Input const& input;
};

} // namespace CSR
