#pragma once

#include "cosyr.h"
#include "input.h"

namespace cosyr {

/**
 * @class Wavelet manager, for both emitted and imported ones.
 *
 */
class Wavelets {
private:

  /**
   * @struct Wavelets data.
   *
   */
  struct Data {
    struct {
      /** coordinates of wavelets on device */
      Kokkos::View<double*[DIM]> coords;
      /** radiation quantities of wavelet on device */
      Kokkos::vector<Kokkos::View<double*>> fields;
    } device;

    struct {
      /** coordinates of wavelets on host */
      Kokkos::View<double*[DIM]>::HostMirror coords;
      /** radiation quantities of wavelets on host */
      Kokkos::vector<Kokkos::View<double*>::HostMirror> fields;
    } host;
  };

  /**
   * @brief Import prescribed wavelets and transfer to device.
   *
   */
  void import_and_transfer();

public:

  /**
   * @brief Create an instance of wavelet manager.
   *
   * @param in_input: simulation parameters
   * @param in_timer: timer.
   */
  explicit Wavelets(Input& in_input, Timer& in_timer);

  /**
   * @brief Delete current instance.
   *
   */
  ~Wavelets() = default;

  /**
   * @brief Copy emitted wavelets to host.
   *
   * @return number of active wavelets per particle.
   */
  Kokkos::View<int*>::HostMirror transfer_to_host();

  /**
   * @brief Number of fields.
   *
   */
  int num_fields = DIM + 1;

  /**
   * @brief Loaded wavelets data.
   *
   */
  Data loaded;

  /**
   * @brief Emitted wavelets data.
   *
   */
  Data emitted;

  /**
   * @brief Count of active wavelets per particle.
   *
   */
  Kokkos::View<int*> active;

  /**
   * @brief Reference to simulation parameters.
   *
   */
  Input& input;

  /**
   * @brief Reference to timer.
   *
   */
  Timer& timer;
};

} // namespace 'cosyr'