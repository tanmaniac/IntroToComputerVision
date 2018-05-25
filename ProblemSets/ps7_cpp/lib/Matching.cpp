#include "../include/Matching.h"

#include <opencv2/ml/ml.hpp>

void makeNaiveCrossValidationSet(const cv::Mat& features,
                                 const cv::Mat& labels,
                                 const size_t removeIdx,
                                 cv::Mat& trainingFeatures,
                                 cv::Mat& trainingLabels,
                                 cv::Mat& inferenceFeatures,
                                 cv::Mat& inferenceLabels) {
    trainingFeatures.create(0, features.cols, features.type());
    trainingLabels.create(0, labels.cols, labels.type());
    inferenceFeatures.create(0, features.cols, features.type());
    inferenceLabels.create(0, labels.cols, labels.type());

    // Iterate over the rows
    for (int r = 0; r < features.rows; r++) {
        if (r == removeIdx) {
            inferenceFeatures.push_back(features.row(r));
            inferenceLabels.push_back(labels.row(r));
            //}
        } else {
            trainingFeatures.push_back(features.row(r));
            trainingLabels.push_back(labels.row(r));
        }
    }
}

void matching::naiveConfusionMatrix(const cv::Mat& features,
                                    const cv::Mat& labels,
                                    cv::Mat& confusion) {
    assert(features.rows == labels.rows);

    auto knn = cv::ml::KNearest::create();

    confusion = cv::Mat::zeros(3, 3, CV_32F);
    cv::Mat counts = cv::Mat::zeros(3, 1, CV_32F); // Counts of each expected label

    cv::Mat trainingFeatures, trainingLabels, inferenceFeatures, inferenceLabels;
    // Iterate over each row of the sample data
    for (int r = 0; r < features.rows; r++) {
        makeNaiveCrossValidationSet(features,
                                    labels,
                                    r,
                                    trainingFeatures,
                                    trainingLabels,
                                    inferenceFeatures,
                                    inferenceLabels);
        // std::cout << "inferenceFeatures= " << inferenceFeatures
        //           << "\ninferenceLabels= " << inferenceLabels << std::endl;
        knn->train(trainingFeatures, cv::ml::ROW_SAMPLE, trainingLabels);
        // cv::Mat results;
        std::vector<float> results;
        knn->findNearest(inferenceFeatures, 3, results);
        // std::cout << "Result = " << results.at<float>(0, 0) << std::endl;
        float result = results[0];
        float expected = inferenceLabels.at<float>(0, 0);
        // std::cout << "Result = " << results[0] << std::endl;
        // Update confusion matrix
        assert (expected >= 1 && result >= 1);
        counts.at<float>(expected - 1, 0) += 1;
        confusion.at<float>(expected - 1, result - 1) += 1;
    }

    // Get percentages by dividing the count of inferred labels by the count of expected labels for
    // each inference
    for (int c = 0; c < confusion.cols; c++) {
        cv::divide(confusion.col(c), counts, confusion.col(c));
    }
}

void makeCrossValidationSet(const cv::Mat& features,
                            const cv::Mat& labels,
                            const cv::Mat& people,
                            const int personToRemove,
                            cv::Mat& trainingFeatures,
                            cv::Mat& trainingLabels,
                            cv::Mat& inferenceFeatures,
                            cv::Mat& inferenceLabels) {
    trainingFeatures.create(0, features.cols, features.type());
    trainingLabels.create(0, labels.cols, labels.type());
    inferenceFeatures.create(0, features.cols, features.type());
    inferenceLabels.create(0, labels.cols, labels.type());

    // Iterate over the rows
    for (int r = 0; r < features.rows; r++) {
        if (personToRemove == people.at<int>(r, 0)) {
            inferenceFeatures.push_back(features.row(r));
            inferenceLabels.push_back(labels.row(r));
            //}
        } else {
            trainingFeatures.push_back(features.row(r));
            trainingLabels.push_back(labels.row(r));
        }
    }
}

void matching::confusionMatrix(const cv::Mat& features,
                               const cv::Mat& labels,
                               const cv::Mat& people,
                               const size_t numPeople,
                               std::vector<cv::Mat>& confusions) {
    assert(features.rows == labels.rows && labels.rows == people.rows);
    assert(labels.cols == 1 && people.cols == 1);

    auto knn = cv::ml::KNearest::create();

    cv::Mat trainingFeatures, trainingLabels, inferenceFeatures, inferenceLabels;
    const cv::Size cfSize(3, 3);

    // Iterate for each person
    for (int p = 1; p <= numPeople; p++) {
        makeCrossValidationSet(features,
                           labels,
                           people,
                           p,
                           trainingFeatures,
                           trainingLabels,
                           inferenceFeatures,
                           inferenceLabels);
        // std::cout << "inferenceFeatures= " << inferenceFeatures
        //           << "\ninferenceLabels= " << inferenceLabels << std::endl;
        knn->train(trainingFeatures, cv::ml::ROW_SAMPLE, trainingLabels);
        // cv::Mat results;
        std::vector<float> results;
        knn->findNearest(inferenceFeatures, 1, results);
        assert(results.size() == inferenceLabels.rows);
        
        // Create output matrix for this person
        cv::Mat confusion = cv::Mat::zeros(cfSize, CV_32F);
        cv::Mat counts = cv::Mat::zeros(3, 1, CV_32F);
        
        // Extract results and add to confusion matrix
        for (int idx = 0; idx < results.size(); idx++) {
            float result = results[idx];
            float expected = inferenceLabels.at<float>(0, idx);
            assert (expected > 0 && result > 0);
            counts.at<float>(expected - 1, 0) += 1;
            confusion.at<float>(expected - 1, result - 1) += 1;
        }
        
        for (int c = 0; c < confusion.cols; c++) {
            cv::divide(confusion.col(c), counts, confusion.col(c));
        }

        confusions.push_back(confusion);
    }

    // Average the confusion matrices;
    cv::Mat avg = cv::Mat::zeros(cfSize, CV_32F);
    for (const auto& confusion : confusions) {
        avg = avg + confusion;
    }
    avg = avg / float(confusions.size());

    confusions.push_back(avg);
}
