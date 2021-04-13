FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="America/Denver"

USER root

RUN apt-get update && \
    apt-get --no-install-recommends install build-essential -y cmake \
    g++ make git ssh-client python3 mpich libmpich-dev \
    python3.8-distutils libboost-dev && \
    apt-get clean

ARG dev="/home/dev"
ARG dependencies="/home/dependencies"
RUN mkdir -p ${dependencies}

# setup SSH
# - create the .ssh directory.
# - populate private key file.
# - set required permissions
# - add github to list of know hosts.
ARG SSH_KEY

RUN mkdir -p /root/.ssh/ && \
    echo "${SSH_KEY}" > /root/.ssh/id_rsa && \
    chmod -R 600 /root/.ssh/ && \
    ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# copy project files
ARG cosyr="/home/cosyr"
RUN mkdir -p ${cosyr}
COPY . ${cosyr}
RUN ls -l ${cosyr}

# install kokkos
WORKDIR ${dev}
RUN git clone git@github.com:kokkos/kokkos.git && \
    cd kokkos && \
    mkdir build && \
    cd build && \
    cmake -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${dependencies}/kokkos" \
      -DKokkos_ENABLE_TESTS=Off \
      -DKokkos_ENABLE_SERIAL=On \
      -DKokkos_ENABLE_OPENMP=On \
      ..  && \
    make -j 4 && \
    make install

# install cabana
WORKDIR ${dev}
RUN git clone git@github.com:ECP-copa/Cabana.git cabana && \
    cd cabana && \
    mkdir build && \
    cd build && \
    cmake -Wno-dev \
      -DCMAKE_BUILD_TYPE="Release" \
      -DCMAKE_PREFIX_PATH="${dependencies}/kokkos" \
      -DCMAKE_INSTALL_PREFIX="${dependencies}/cabana" \
      -DCabana_REQUIRE_OPENMP=ON \
      -DCabana_ENABLE_MPI=ON \
      -DCabana_ENABLE_EXAMPLES=ON \
      -DCabana_ENABLE_TESTING=OFF \
      -DCabana_ENABLE_PERFORMANCE_TESTING=OFF \
      -DCabana_ENABLE_CAJITA=ON \
      .. && \
    make -j 4 && \
    make install

# install pybind
WORKDIR ${dev}
RUN git clone git@github.com:pybind/pybind11.git pybind && \
    cd pybind && \
    mkdir build && \
    cd build && \
    cmake -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${PYTHONPATH}" \
      -DCMAKE_INSTALL_PREFIX="${dependencies}/pybind" \
      -DPYBIND11_TEST=Off \
      .. && \
    make -j 4 && \
    make install

# install thrust
WORKDIR ${dev}
RUN git clone git@github.com:NVIDIA/thrust.git "${dependencies}/thrust"

# install wonton
WORKDIR ${dev}
RUN git clone --recursive git@github.com:laristra/wonton.git && \
    cd wonton && \
    git checkout 1.2.11 && \
    mkdir build && \
    cd build && \
    cmake -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${dependencies}/wonton" \
      -DENABLE_UNIT_TESTS=False \
      -DWONTON_ENABLE_MPI=True \
      -DWONTON_ENABLE_THRUST=True \
      -DWONTON_ENABLE_LAPACKE=False \
      -DTHRUST_ROOT="${dependencies}/thrust" \
      .. && \
    make -j 4 && \
    make install

# install portage
WORKDIR ${dev}
RUN git clone --recursive git@github.com:laristra/portage.git && \
    cd portage && \
    mkdir build && \
    cd build && \
    ls -al "${dependencies}/wonton" && \
    cmake -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${dependencies}/portage" \
      -DENABLE_UNIT_TESTS=False \
      -DPORTAGE_ENABLE_MPI=True \
      -DPORTAGE_ENABLE_Jali=False \
      -DPORTAGE_ENABLE_FleCSI=False \
      -DPORTAGE_ENABLE_THRUST=True \
      -DWONTON_ROOT="${dependencies}/wonton" \
      .. && \
    make -j 4 && \
    make install

RUN ls -l "${dependencies}"

ENTRYPOINT ["server"]