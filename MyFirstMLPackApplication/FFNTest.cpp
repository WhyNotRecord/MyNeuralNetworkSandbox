#pragma once

#include "FFNTest.h"
#include "Utils.h"
#include <mlpack/mlpack.hpp>
#include <filesystem>

using namespace mlpack;

int testClassification() {//TODO split into train and test blocks, save trained model, load model istead of training if saved one exists
    //Load train data
    std::string trainFile = std::string("thyroid_train.csv");
    if (!std::filesystem::exists(trainFile))
        return 1;
    arma::mat trainData;
    data::Load(trainFile, trainData, true);

    // Split the labels from the training set and testing set respectively.
    // Decrement the labels by 1, so they are in the range 0 to (numClasses - 1).
    arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
    trainData.shed_row(trainData.n_rows - 1);

    // Initialize the network.
    FFN<> model;
    //FFN<mlpack::NegativeLogLikelihood, mlpack::HeInitialization> model;
    model.Add<Linear>(8);
    model.Add<Sigmoid>();
    model.Add<Linear>(3);
    model.Add<LogSoftMax>();

    //ens::AdamType<ens::AdamUpdate> optimizer;
    ens::RMSProp optimizer;//default
    // Train the model.
    model.Train(trainData, trainLabels, optimizer,
        ens::PrintLoss(),
        ens::ProgressBar());

    //Load test data
    std::string testFile = std::string("thyroid_test.csv");
    if (!std::filesystem::exists(testFile))
        return 2;
    arma::mat testData;
    data::Load(testFile, testData, true);

    arma::mat testLabels = testData.row(testData.n_rows - 1) - 1;
    testData.shed_row(testData.n_rows - 1);

    // Use the Predict method to get the predictions.
    arma::mat predictionTemp;
    model.Predict(testData, predictionTemp);

    /*
    Since the predictionsTemp is of dimensions (3 x number_of_data_points)
    with continuous values, we first need to reduce it to a dimension of
    (1 x number_of_data_points) with scalar values, to be able to compare with
    testLabels.

    The first step towards doing this is to create a matrix of zeros with the
    desired dimensions (1 x number_of_data_points).

    In predictionsTemp, the 3 dimensions for each data point correspond to the
    probabilities of belonging to the three possible classes.
    */
    int cols = predictionTemp.n_cols;
    arma::mat prediction = arma::zeros<arma::mat>(1, cols);

    predictionTemp.brief_print();
    // Find index of max prediction for each data point and store in "prediction"
    for (size_t i = 0; i < cols; ++i)
    {
        auto m = arma::max(predictionTemp.col(i));
        auto f = arma::find(m == predictionTemp.col(i), 1);//returns 1 index of i-th column, that equals m
        prediction(i) = arma::as_scalar(f);
    }
    prediction.brief_print();
    testLabels.brief_print();

    /*
    Compute the error between predictions and testLabels,
    now that we have the desired predictions.
    */
    size_t correct = arma::accu(prediction == testLabels);
    double classificationError = 1 - double(correct) / testData.n_cols;
    // Print out the classification error for the testing dataset.
    std::cout << "Classification Error for the Test set: " << classificationError << std::endl;
    return 0;
}

int myTestClassification() {
    // Load the training set and testing set.
    auto sets = loadDataSetAlt("winequality-white.csv");
    arma::mat trainData = std::get<0>(sets);
    arma::mat testData = std::get<1>(sets);

    // Split the labels from the training set and testing set respectively.
    // Decrement the labels by 1, so they are in the range 0 to (numClasses - 1).
    arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
    arma::mat testLabels = testData.row(testData.n_rows - 1) - 1;
    trainData.shed_row(trainData.n_rows - 1);
    testData.shed_row(testData.n_rows - 1);

    mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::HeInitialization> model;
    model.Add<Linear>(5);
    model.Add<Linear>(10);
    model.Add<LogSoftMax>();

    ens::AdamType<ens::AdamUpdate> optimizer;
    //ens::SGD optimizer;
    //ens::RMSProp optimizer;//default
    // optimizer.Tolerance() = 0.0000001;
    // MaxIterations and BatchSize regulate number of epochs
    //optimizer.MaxIterations() = 1000000;
    //optimizer.BatchSize() = 50 ;
    // Train the model.
    model.Train(trainData, trainLabels, optimizer,
        ens::PrintLoss(),
        ens::ProgressBar()/*,
        ens::EarlyStopAtMinLoss(),
        ens::StoreBestCoordinates<arma::mat>()*/);

    return 0;
}