#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "grpc_client.h"

#define FAIL_IF_ERR(X)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace tc = triton::client;

std::string GetEnvStr(std::string const &key)
{
    char *val = getenv(key.c_str());
    return val == NULL ? std::string("") : std::string(val);
}

int main()
{
    // Config and some hard coded names, assuming that it is always a model
    // of same kind.
    std::string stream_url = GetEnvStr("STREAM");
    std::string server_url = GetEnvStr("SERVER");
    std::string model_name = GetEnvStr("MODEL_NAME");
    const std::string input_name = "images";
    const std::string output_name = "output";
    std::vector<int64_t> input_shape = {1, 112, 112, 3};

    // Create a client
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, server_url, true)
    );

    // Connect with video stream
    cv::VideoCapture stream(stream_url);

    // Read frame
    cv::Mat frame;
    stream >> frame;
    if (frame.empty())
    {
        std::cout << "Failed" << std::endl;
        return -1;
    }

    // Preprocess image
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_shape[2], input_shape[1]));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    int data_size = input_shape[1] * input_shape[2] * input_shape[3];
    float *data = new float[data_size];
    for (int i = 0; i < data_size; i++)
    {
        data[i] = (float) resized.at<uint8_t>(i);
    }

    // Create input
    tc::InferInput* input;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input, input_name, input_shape, "FP32")
    );
    FAIL_IF_ERR(
        input->AppendRaw(resized.data, data_size * sizeof(float))
    );
    std::shared_ptr<tc::InferInput> input_ptr(input);
    std::vector<tc::InferInput*> inputs = {input_ptr.get()};

    // Create output
    tc::InferRequestedOutput* output;
    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output, "output")
    );
    std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};

    // Send request
    tc::InferOptions options(model_name);
    tc::InferResult* result;
    FAIL_IF_ERR(
        client->Infer(&result, options, inputs, outputs)
    );
    std::shared_ptr<tc::InferResult> result_ptr(result);

    // Print results
    std::vector<int64_t> output_shape;
    FAIL_IF_ERR(result->Shape(output_name, &output_shape));
    // float* data_ptr;
    // FAIL_IF_ERR(result->RawData(output_name, ...

    // Result isn't meaningful at all, so why bother

    return 0;
}
