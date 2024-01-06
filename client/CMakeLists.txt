cmake_minimum_required (VERSION 3.18)

add_executable(client client.cpp)
target_include_directories(
    client
    PRIVATE
    /usr/local/include/opencv4
    /workspace/triton/include
)
target_link_directories(
    client
    PRIVATE
    /usr/local/lib
    /workspace/triton/lib
)
target_link_libraries(
    client
    PRIVATE
    opencv_core
    opencv_imgproc
    opencv_videoio
    grpcclient
    pthread
)
