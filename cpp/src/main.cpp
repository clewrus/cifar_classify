#include "utils.cpp"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

std::vector<std::string> getInputFiles(const std::string &input) {
    std::vector<std::string> inputFiles;

    if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (entry.is_regular_file()) {
                inputFiles.push_back(entry.path().string());
            }
        }
    } else {
        inputFiles.push_back(input);
    }

    return inputFiles;
}

int main(int argc, char* argv[]) {
    ArgParser args;
    try {
        args.parse(argc, argv);
        args.validate();
    } catch (std::runtime_error e) {
        std::cout << "Runtime exception cought: " << e.what() << std::endl;
        return -1;
    }

    ModelWrapper model;

    try {
        model.initialize(args.getModel());
    } catch (Ort::Exception oe) {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << std::endl;
        return -1;
    }

    std::vector<float> prob;
    std::vector<std::string> labels;

    std::vector<std::string> inputFiles = getInputFiles(args.getInput());
    for (const std::string &s : inputFiles) {
        try {
            model.runOnImage(s, prob, labels);
        } catch (Ort::Exception oe) {
            std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << std::endl;
            return -1;
        } catch (std::runtime_error e) {
            std::cout << "Runtime exception cought: " << e.what() << std::endl;
            return -1;
        }
        
        for (size_t i = 0; i < args.getTopK(); ++i) {
            std::cout << "class: " << labels[i] << ", probability: " << prob[i] << std::endl;
        }

        std::cout << std::endl;
    }

    return 0;
}
