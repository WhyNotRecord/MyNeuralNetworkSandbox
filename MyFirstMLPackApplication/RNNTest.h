#pragma once

#include <mlpack/mlpack.hpp>

void GenerateNoisySines(arma::cube& data,
    arma::cube& labels,
    size_t rho,
    const size_t dataPoints = 100,
    const double noisePercent = 0.2);

int testPrediction();