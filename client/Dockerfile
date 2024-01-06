FROM nvcr.io/nvidia/tritonserver:23.05-py3-sdk

RUN apt-get install -y build-essential cmake
WORKDIR /workspace/app
COPY . .

WORKDIR /workspace/app/build
RUN cmake ..
RUN make

CMD ./client
