FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y g++ gcc make cmake wget unzip tar pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
WORKDIR /workspace/opencv
RUN wget -q https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip \
    && unzip -q 4.7.0.zip
WORKDIR /workspace/opencv/build
RUN cmake \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_LIST=core,videoio,imgproc \
    ../opencv-4.7.0
RUN make -j$(nproc)
RUN make install
RUN ldconfig

WORKDIR /workspace/triton
RUN wget -q https://github.com/triton-inference-server/server/releases/download/v2.34.0/v2.34.0_ubuntu2004.clients.tar.gz
RUN tar zxf v2.34.0_ubuntu2004.clients.tar.gz

WORKDIR /workspace/app
COPY . .

WORKDIR /workspace/app/build
RUN cmake ..
RUN make

CMD ./client
