# Evaluation Task
## Pre-requisites
Place ONNX model file at `model/resnet50/1/model.onnx`.

## Docker Images / Tools
- NVIDIA Triton server.
- NVIDIA Triton client. This image is extended for custom client, and comes with Triton client and OpenCV pre-installed.
- `tiangolo/nginx-rtmp` used for RTMP server.

## How to run
To get triton server and nginx server up,
```
docker compose up -d tritonserver nginx-rtmp
```
To run client,
```
docker compose up --build client
```
