# Evaluation Task
## Pre-requisites
Place ONNX model file at `model/resnet50/1/model.onnx`.

## Docker Images / Tools
- NVIDIA Triton server.
- `tiangolo/nginx-rtmp` used for RTMP server.
- Client image is based upon ubuntu. It builds OpenCV, download Triton client library and makes client code. 

## How to run
To get triton server and nginx server up,
```
docker compose up -d tritonserver nginx-rtmp
```
Start video streaming. You can do it using OBS Studio setting server to `rtmp://localhost:1935/live` and key to `stream` (depending upon env). To run client,
```
docker compose up --build client
```

# Code Explanation
The client creates a streaming threads that gets frames from the RTML stream, pushes them into the queue and post the semaphore. The processing threads wait for the semaphore to be posted, and then pops the frame from the queue and processes it. The queue is made thread safe using a mutex.

Once frame a is popped, it is cropped, resized down to model size and converted to a float tensor. This float tensor is passed to Triton server, and response is received. The cycle continues. Most of the pointers are wrapped in `unique_ptr` and `shared_ptr` to avoid memory leaks.

CMake is used to build the client code. CMake links OpenCV and Triton client libraries when building the client code.