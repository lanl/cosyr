#!/bin/bash

docker pull hobywan/cosyr
docker run -v "$(pwd):/home/cosyr" -it hobywan/cosyr /bin/bash -c \
      "cd /home/cosyr && \
      mkdir build-test &&
      cd build-test && \
      cmake -Wno-dev \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_TCMALLOC=False \
        -DPORTAGE_DIR=/home/dependencies/portage \
        -DKokkos_DIR=/home/dependencies/kokkos \
        -Dpybind11_DIR=/home/dependencies/pybind \
        -DCabana_DIR=/home/dependencies/cabana \
        .. && \
      make -j 4"