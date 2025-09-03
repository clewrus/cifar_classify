#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>  
#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;


constexpr unsigned int IMG_SIZE = 32;
const std::array<std::string, 10> CIFAR_LABELS = {
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
};

class ArgParser {
public:
    explicit ArgParser() {}

    const std::string& getModel() const { return modelPath; }
    const std::string& getInput() const { return inputPath; }
    int getTopK() const { return topK; }

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--model") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --model");
                modelPath = std::string(argv[++i]);
            }
            else if (arg == "--input") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --input");
                inputPath = std::string(argv[++i]);
            }
            else if (arg == "--topK") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --topK");
                topK = std::stoi(argv[++i]);
            }
            else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        if (modelPath.empty()) throw std::runtime_error("--model is required");
        if (inputPath.empty()) throw std::runtime_error("--input is required");
    }

    void validate() const {
        if (!fs::exists(modelPath)) {
            throw std::runtime_error("Model file does not exist: " + modelPath);
        }
        if (!fs::exists(inputPath)) {
            throw std::runtime_error("Input file does not exist: " + inputPath);
        }
        if (topK < 1 || topK > 10) {
            throw std::runtime_error("topK must be between 1 and 10");
        }
    }

private:
    std::string modelPath;
    std::string inputPath;
    int topK = 1;
};

class ModelWrapper {
public:
    explicit ModelWrapper() {};

    void initialize(const std::string &modelPath) {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);

        std::wstring wmodelPath(modelPath.begin(), modelPath.end());
        session = std::make_unique<Ort::Session>(env, wmodelPath.c_str(), sessionOptions);

        Ort::TypeInfo info = session->GetInputTypeInfo(0);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        inputDim = tensor_info.GetShape();
        inputDim[0] = 1;

        inputName = session->GetInputNames().front();
        outputName = session->GetOutputNames().front();
    }

    cv::Mat preprocessImage(const std::string &imgPath) {
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            throw std::runtime_error("Failed to load image\n");
        }

        cv::Mat prepImg = cv::dnn::blobFromImage(img, 1. / 255., {IMG_SIZE, IMG_SIZE}, 0, true, false, CV_32F);
        return prepImg;
    }

    void runOnImage(const std::string &imgPath, std::vector<float> &prob, std::vector<std::string> &labels) {
        cv::Mat prepImg = preprocessImage(imgPath);
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memoryInfo, 
            reinterpret_cast<float*>(prepImg.data),
            prepImg.total(),
            inputDim.data(), 
            inputDim.size()
        );

        std::vector<const char *> inputNames {inputName.c_str()};
        std::vector<const char *> outputNames {outputName.c_str()};
        std::vector<Ort::Value> outputTensors = session->Run(
            Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outputNames.data(), 1
        );

        if (outputTensors.empty()) {
            throw std::runtime_error("No output tensor");
        }

        const Ort::Value& probTensor = outputTensors.front();

        auto shapeInfo = probTensor.GetTensorTypeAndShapeInfo();
        if (shapeInfo.GetElementCount() != CIFAR_LABELS.size()) {
            std::stringstream s;
            s << "Unexpected output dimension. Expected " << CIFAR_LABELS.size() << ". Got " << probTensor.GetCount();
            throw std::runtime_error(s.str());
        }

        std::vector<int> idx(CIFAR_LABELS.size());
        std::iota(idx.begin(), idx.end(), 0);

        const float* data = probTensor.GetTensorData<float>();
        std::sort(idx.begin(), idx.end(), [&data](int l, int r) { return data[l] > data[r]; });

        prob.resize(CIFAR_LABELS.size());
        labels.resize(CIFAR_LABELS.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            prob[i] = data[idx[i]];
            labels[i] = CIFAR_LABELS[idx[i]];
        }
    }

private:
    std::unique_ptr<Ort::Session> session;

    std::vector<int64_t> inputDim;
    std::string inputName;
    std::string outputName;
};
