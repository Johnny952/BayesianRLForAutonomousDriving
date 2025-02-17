FROM ubuntu:18.04
RUN apt-get update \
   && apt-get install -y git cmake python3.7 python3-pip g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev mesa-utils
WORKDIR /app
RUN git clone --depth 1 --branch v1_15_0 https://github.com/eclipse/sumo \
   && mkdir sumo/build/cmake-build \
   && cd sumo/build/cmake-build \
   && cmake ../.. \
   && make -j$(nproc) \
   && make install \
   && cd ../../.. \
   && rm -r sumo
ENV SUMO_HOME "/usr/local/share/sumo"
COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV SUMO_USER johnny
ENV LIBGL_ALWAYS_INDIRECT=1
RUN adduser $SUMO_USER --disabled-password
# COPY . .
