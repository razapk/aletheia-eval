#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <queue>

#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>

#include "grpc_client.h"

#define FAIL_IF_ERR(X)                                  \
    {                                                   \
        tc::Error err = (X);                            \
        if (!err.IsOk())                                \
        {                                               \
            std::cerr << "error: " << err << std::endl; \
            exit(1);                                    \
        }                                               \
    }

namespace tc = triton::client;

// I have checked. Queue works well with cv::Mat in terms of memory management.
typedef std::queue<cv::Mat> ImageQueue;

struct ThreadConfig
{
    ImageQueue queue;
    pthread_mutex_t mux;
    sem_t sem;
};

std::string GetEnvStr(const char *key)
{
    char *val = getenv(key);
    return val == NULL ? std::string("") : std::string(val);
}

int GetEnvInt(const char *key, int fallback = -1)
{
    char *val = getenv(key);
    return val == NULL ? fallback : atoi(val);
}

// Thread function to process video using Triton
void *ProcessingThread(void *ptr)
{
    // Config and some hard coded names, assuming that it is always a model
    // of same kind.
    std::string server_url = GetEnvStr("SERVER");
    std::string model_name = GetEnvStr("MODEL_NAME");
    const std::string input_name = "images";
    const std::string output_name = "output";
    std::vector<int64_t> input_shape = {1, 112, 112, 3};

    // Get thread id
    pthread_t thread_id = pthread_self();

    // Create a client
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(tc::InferenceServerGrpcClient::Create(&client, server_url, true));

    // Loop to get and process images
    ThreadConfig *config = (ThreadConfig *)ptr;
    while (true)
    {
        // Get a frame
        sem_wait(&config->sem);
        pthread_mutex_lock(&config->mux);
        cv::Mat frame = config->queue.back();
        config->queue.pop();
        pthread_mutex_unlock(&config->mux);

        // See, I can crop image
        cv::Mat cropped = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows / 2));

        // Preprocess image
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(input_shape[2], input_shape[1]));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        cv::Mat typed;
        resized.convertTo(typed, CV_32FC3);

        int byte_size = input_shape[1] * input_shape[2] * input_shape[3] * sizeof(float);
        std::vector<uint8_t> buffer;
        buffer.resize(byte_size);
        memcpy(&(buffer[0]), typed.datastart, byte_size);

        // Create input
        tc::InferInput *input;
        FAIL_IF_ERR(
            tc::InferInput::Create(&input, input_name, input_shape, "FP32"));
        FAIL_IF_ERR(input->AppendRaw(buffer));
        std::shared_ptr<tc::InferInput> input_ptr(input);
        std::vector<tc::InferInput *> inputs = {input_ptr.get()};

        // Create output
        tc::InferRequestedOutput *output;
        FAIL_IF_ERR(
            tc::InferRequestedOutput::Create(&output, "output"));
        std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);
        std::vector<const tc::InferRequestedOutput *> outputs = {output_ptr.get()};

        // Send request
        tc::InferOptions options(model_name);
        tc::InferResult *result;
        FAIL_IF_ERR(
            client->Infer(&result, options, inputs, outputs));
        std::shared_ptr<tc::InferResult> result_ptr(result);

        // Print results
        std::vector<int64_t> output_shape;
        FAIL_IF_ERR(result->Shape(output_name, &output_shape));
        // float* data_ptr;
        // FAIL_IF_ERR(result->RawData(output_name, ...

        // Result isn't meaningful at all, so why bother
        std::cout << "Successful processing by " << thread_id << std::endl;
    }
    return NULL;
}

// Thread function to handle video stream
void *StreamingThread(void *ptr)
{
    // Get environment variables
    std::string url = GetEnvStr("STREAM");
    int max_size = GetEnvInt("MAX_QUEUE_SIZE", 30);

    // Setup up stream and other suff
    ThreadConfig *config = (ThreadConfig *)ptr;
    cv::VideoCapture capture(url);

    // Start infinite capture loop
    while (true)
    {
        cv::Mat frame;
        // Get a frame
        if (!capture.read(frame))
        {
            // Wait for a second before retrying
            std::cout << "Error" << std::endl;
            sleep(1);
            continue;
        }

        // Lock queue and insert frame
        pthread_mutex_lock(&config->mux);
        int count = 0;
        while (config->queue.size() >= max_size)
        {
            config->queue.pop();
            count--;
        }
        config->queue.push(frame);
        count++;
        pthread_mutex_unlock(&config->mux);

        // Post semaphore
        while (count > 0)
        {
            sem_post(&config->sem);
            count--;
        }
    }
    return NULL;
}

int main()
{
    // Get env
    int num_workers = GetEnvInt("WORKERS", 1);

    // Thread config
    ThreadConfig config;
    pthread_mutex_init(&config.mux, NULL);
    sem_init(&config.sem, 0, 0);

    // Create threads
    int num_threads = num_workers + 1;
    std::cout << "Starting " << num_threads << " threads." << std::endl;
    pthread_t *threads = new pthread_t[num_threads];
    pthread_create(&threads[0], NULL, StreamingThread, (void *)&config);
    for (int i = 1; i < num_threads; i++)
    {
        pthread_create(&threads[i], NULL, ProcessingThread, (void *)&config);
        std::cout << "Created worker thread " << threads[i] << std::endl;
    }

    // Wait for threads
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    delete[] threads;
    return 0;
}
