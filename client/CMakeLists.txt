cmake_minimum_required (VERSION 3.18)

add_executable(client client.cpp)
target_include_directories(
    client
    PRIVATE
    /usr/include/opencv4
    /workspace/install/include
)
target_link_directories(
    client
    PRIVATE
    /workspace/install/lib
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
