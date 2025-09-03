#include "utils.cpp"
#include <gtest/gtest.h>


const std::string catImage = std::string(SAMPLES_DIR) + "/cat.jpg";
const std::string automobileImage = std::string(SAMPLES_DIR) + "/automobile.jpg";
const std::string modelPath = std::string(MODELS_DIR) + "/model.onnx";

TEST(TestPreprocessing, Normalize01) {
    ModelWrapper model;
    cv::Mat prep = model.preprocessImage(catImage);
    double minVal, maxVal;
    cv::minMaxIdx(prep, &minVal, &maxVal);

    EXPECT_GE(minVal, 0.);
    EXPECT_LE(maxVal, 1.);
}

TEST(TestPreprocessing, ResizeTo1x3x32x32) {
    fs::path temp = fs::path(catImage).parent_path() / "tmp.jpg";
    cv::Mat cat = cv::imread(catImage);
    cv::Mat bigCat;
    cv::resize(cat, bigCat, {100, 100});
    cv::imwrite(temp.string().c_str(), bigCat);

    ModelWrapper model;
    cv::Mat prep = model.preprocessImage(temp.string());
    EXPECT_EQ(prep.size[0], 1);
    EXPECT_EQ(prep.size[1], 3);
    EXPECT_EQ(prep.size[2], 32);
    EXPECT_EQ(prep.size[3], 32);
}

TEST(TestInference, ModelInitializes) {
    ModelWrapper model;
    EXPECT_NO_THROW(model.initialize(modelPath));
}

TEST(TestInference, ProbabilitiesAddsUpTo1) {
    ModelWrapper model;
    EXPECT_NO_THROW(model.initialize(modelPath));

    std::vector<float> probs;
    std::vector<std::string> labels;
    EXPECT_NO_THROW(model.runOnImage(catImage, probs, labels));

    float sum = 0;
    std::for_each(probs.begin(), probs.end(), [&sum](float v){sum += v;});
    EXPECT_NEAR(sum, 1, 1e-4);
}

TEST(TestInference, ExpectMeaningfulLabels) {
    ModelWrapper model;
    EXPECT_NO_THROW(model.initialize(modelPath));

    std::vector<float> probs;
    std::vector<std::string> labels;

    EXPECT_NO_THROW(model.runOnImage(catImage, probs, labels));
    EXPECT_EQ(labels.front(), "cat");

    EXPECT_NO_THROW(model.runOnImage(automobileImage, probs, labels));
    EXPECT_EQ(labels.front(), "automobile");
}
