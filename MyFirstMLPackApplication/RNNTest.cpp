#pragma once

#include "RNNTest.h"

using namespace ens;
using namespace mlpack;

/**
 * Generates noisy sine wave and outputs the data and the labels that
 * can be used directly for training and testing with RNN.
 */
void GenerateNoisySines(arma::cube& data,
    arma::cube& labels,
    size_t rho,
    const size_t dataPoints,
    const double noisePercent)
{
    size_t points = dataPoints;
    size_t r = dataPoints % rho;

    if (r == 0)
        points += 1;
    else
        points += rho - r + 1;

    arma::colvec x(points);
    int i = 0;
    double interval = 0.6 / points;
    
    //sets random seed
    //RandomSeed(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch() / 1000).count());
    //generating sin values
    x.for_each([&i, noisePercent, interval]
    (arma::colvec::elem_type& val) {
            double t = interval * (++i);
            val = ::sin(2 * M_PI * 10 * t) + (noisePercent * Random(0.0, 0.1));
        });

    arma::colvec y = x;
    std::cout << "max value before normalization " << y.max() << std::endl;
    y = arma::normalise(x);
    y.raw_print(std::cout, "");

    // Now break this into columns of rho size slices.
    //size_t numColumns = y.n_elem / rho;
    size_t numColumns = y.n_elem - rho - 1;
    data = arma::cube(1, numColumns, rho);
    labels = arma::cube(1, numColumns, 1);

    //for (size_t i = 0; i < numColumns; ++i)
    for (size_t i = 0; i < numColumns; ++i)
    {
        std::cout << "column " << i << std::endl;
        //data.tube(0, i) = y.rows(i * rho, i * rho + rho - 1);
        data.tube(0, i) = y.rows(i, i + rho - 1);
        data.tube(0, i).brief_print(std::cout, "");
        labels.subcube(0, i, 0, 0, i, 0) =
            y.rows(i + rho, i + rho);
            //y.rows(i * rho + rho, i * rho + rho);
        labels.subcube(0, i, 0, 0, i, 0).brief_print(std::cout, "");
    }
}

int testPrediction()
{
    const size_t rho = 10;
    const size_t dataPoints = 100;

    // Generate 12 (2 * 6) noisy sines. A single sine contains rho
    // points/features.
    arma::cube input, labels;
    GenerateNoisySines(input, labels, rho, dataPoints, 1);

    /**
     * Construct a network with 1 input unit, 4 LSTM units and 1 output
     * unit. The hidden layer is connected to itself. The network structure
     * looks like:
     *
     *  Input         Hidden        Output
     * Layer(1)      LSTM(4)       Layer(1)
     * +-----+       +-----+       +-----+
     * |     |       |     |       |     |
     * |     +------>|     +------>|     |
     * |     |    ..>|     |       |     |
     * +-----+    .  +--+--+       +-----+
     *            .     .
     *            .     .
     *            .......
     *
     * We use MeanSquaredError for the loss type, since we are predicting a
     * continuous value.
     */
    RNN<MeanSquaredError> model(rho, true /* only one response per sequence */);
    model.Add<LSTM>(4);
    model.Add<LinearNoBias>(1);

    arma::cube trainInput = input.subcube(0, 0, 0, input.n_rows - 1, input.n_cols - rho - 1, input.n_slices - 1), 
        trainLabels = labels.cols(0, labels.n_cols - rho - 1);
    StandardSGD opt(0.1, 1, 10 * trainInput.n_cols /* 10 epochs */, -100);
    model.Train(trainInput, trainLabels, opt);

    // Now compute the MSE on the training set.
    arma::cube trainPredictions;
    model.Predict(trainInput, trainPredictions);
    const double mse = arma::accu(arma::square(
        arma::vectorise(trainLabels) -
        arma::vectorise(trainPredictions.slice(trainPredictions.n_slices - 1)))) /
        trainInput.n_cols;
    std::cout << "MSE on training set is " << mse << "." << std::endl;

    arma::cube fullPredictions;
    model.Predict(input, fullPredictions);
    const double fullMse = arma::accu(arma::square(
        arma::vectorise(labels) -
        arma::vectorise(fullPredictions.slice(fullPredictions.n_slices - 1)))) /
        input.n_cols;
    std::cout << "MSE on full set is " << fullMse << "." << std::endl;

    arma::cube randPredictions = arma::randn(1, dataPoints - rho, 1);
    double rMax = randPredictions.max();
    std::cout << "max value of random distribution before normalization " << rMax << std::endl;
    randPredictions = randPredictions /= rMax;
    //(1, dataPoints - rho, 1, arma::fill::fill_randn);
    const double randMse = arma::accu(arma::square(
        arma::vectorise(labels) -
        arma::vectorise(randPredictions.slice(randPredictions.n_slices - 1)))) /
        input.n_cols;
    std::cout << "MSE on random set is " << randMse << "." << std::endl;

    return 0;
}