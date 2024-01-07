# Evaluation Task - Improved Version

## Pre-requisites
Ensure that the ONNX model file is located at `model/resnet50/1/model.onnx`.

## Docker Images / Tools
- NVIDIA Triton server.
- The `tiangolo/nginx-rtmp` image is utilized for the RTMP server.
- The client image is based on Ubuntu, incorporating OpenCV build, Triton client library download, and client code compilation.

## How to Run
1. Start Triton server and NGINX server:
    ```bash
    docker compose up -d tritonserver nginx-rtmp
    ```
2. Initiate video streaming using OBS Studio, configuring the server to `rtmp://localhost:1935/live` and the key to `stream` (depending on the environment).
3. Run the client:
    ```bash
    docker compose up --build client
    ```

# Code Explanation
The client operates with streaming threads that capture frames from the RTMP stream. These frames are then pushed into a queue, followed by posting a semaphore. The processing threads, in turn, wait for the semaphore to be posted, extract the frame from the queue, and proceed with the necessary processing. The queue is designed to be thread-safe through the use of a mutex.

Upon popping a frame, the client performs cropping, resizes it to the model size, and converts it into a float tensor. This float tensor is subsequently sent to the Triton server, and the response is received. This cycle continues. Pointers are intelligently managed using `unique_ptr` and `shared_ptr` to prevent memory leaks.

CMake is employed for building the client code, with proper linkage to OpenCV and Triton client libraries during the compilation process.