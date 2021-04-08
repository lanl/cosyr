#include "wavelet.h"

namespace cosyr {

/* -------------------------------------------------------------------------- */
Wavelets::Wavelets(Input& in_input, Timer& in_timer)
  : input(in_input),
    timer(in_timer)
{
  int const num_emitted = input.kernel.num_particles
                        * input.kernel.num_wavefronts
                        * input.kernel.num_dirs;

  Kokkos::resize(active, input.kernel.num_particles);
  Kokkos::resize(emitted.device.coords, num_emitted);
  Kokkos::resize(emitted.device.fields, num_fields);
  Kokkos::resize(emitted.host.fields, num_fields);
  emitted.host.coords = Kokkos::create_mirror_view(emitted.device.coords);

  for (int k = 0; k < num_fields; ++k) {
    Kokkos::resize(emitted.device.fields[k], num_emitted);
    emitted.host.fields[k] = Kokkos::create_mirror_view(emitted.device.fields[k]);
  }

  if (input.wavelets.found and input.wavelets.subcycle) {
    int const num_loaded = input.kernel.num_particles * input.wavelets.count;

    Kokkos::resize(loaded.device.coords, num_loaded);
    Kokkos::resize(loaded.device.fields, num_fields);
    Kokkos::resize(loaded.host.fields, num_fields);
    loaded.host.coords = Kokkos::create_mirror_view(loaded.device.coords);

    for (int k = 0; k < num_fields; ++k) {
      Kokkos::resize(loaded.device.fields[k], num_loaded);
      loaded.host.fields[k] = Kokkos::create_mirror_view(loaded.device.fields[k]);
    }

    import_and_transfer();
  }
}

/* -------------------------------------------------------------------------- */
void Wavelets::import_and_transfer() {

  if (input.wavelets.found and input.wavelets.subcycle) {
    timer.start("import_wavelet");

    for (int i = 0; i < input.kernel.num_particles; i++) {
      int const start = i * input.wavelets.count;

      for (int j = 0; j < input.wavelets.count; j++) {
        int const k = j + start;
        loaded.host.coords(k, WT_POS_X) = input.wavelets.x[j];
        loaded.host.coords(k, WT_POS_Y) = input.wavelets.y[j];
        #if DIM==3
          loaded.host.coords(k, WT_POS_Z) = input.wavelets.z[j];
        #endif

        for (int f = 0; f < num_fields; ++f) {
          int const offset = (f * input.wavelets.count);
          loaded.host.fields[f](k) = input.wavelets.field[j + offset];
        }
      }
      timer.stop("import_wavelet");
      timer.start("copy_wavelet_to_device");

      int const extent  = start + input.wavelets.count;
      auto range_coords = std::make_pair(start * DIM, extent * DIM);
      auto range_fields = std::make_pair(start, extent);

      Kokkos::deep_copy(subview(loaded.device.coords, range_coords, Kokkos::ALL()),
                        subview(loaded.host.coords, range_coords, Kokkos::ALL()));

      for (int f = 0; f < num_fields; f++) {
        Kokkos::deep_copy(subview(loaded.device.fields[f], range_fields),
                          subview(loaded.host.fields[f], range_fields));
      }
      timer.stop("copy_wavelet_to_device");
    }
  }
}

/* -------------------------------------------------------------------------- */
Kokkos::View<int*>::HostMirror Wavelets::transfer_to_host() {

  // TODO: no need to copy in the future when Portage is ported to GPU/Kokkos
  timer.start("wavelet_to_host");

  auto num_active = Kokkos::create_mirror_view(active);

  Kokkos::deep_copy(num_active, active);
  Kokkos::deep_copy(emitted.host.coords, emitted.device.coords);

  for (int i = 0; i < num_fields; ++i) {
    Kokkos::deep_copy(emitted.host.fields[i], emitted.device.fields[i]);
  }

  timer.stop("wavelet_to_host");

  return num_active;
}

/* -------------------------------------------------------------------------- */
} // namespace 'cosyr'