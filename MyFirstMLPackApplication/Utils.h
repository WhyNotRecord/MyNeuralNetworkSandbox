#pragma once

//To enable serialization for neural networks, define the MLPACK_ENABLE_ANN_SERIALIZATION macro before including mlpack
//#define MLPACK_ENABLE_ANN_SERIALIZATION

//#include <tuple>
#include <mlpack/mlpack.hpp>

template<typename eT> void normal_print(std::ostream& o, const arma::Mat<eT> &m, bool print_size);

std::tuple<arma::mat, arma::mat> loadDataSet(std::string filename, float testDataBound = 0.75f);

std::tuple<arma::mat, arma::mat> loadDataSetAlt(std::string filename, float testDataBound = 0.75f, int crossLen = 0);

std::tuple<arma::cube, arma::cube> loadDataSetCube(std::string filename, float testDataBound = 0.75f);

std::tuple<arma::cube, arma::cube> preparePredictionCubes(arma::mat &valuesMatrix, int analCount = 100, int predictCount = 3, bool print = false);

std::tuple<arma::cube, arma::cube> preparePredictionCubes(arma::mat &valuesMatrix, int inputCount, int analCount = 100, int predictCount = 1, bool print = false);

std::array<double, 3> calculateDifferencePrecents(arma::cube& c1, arma::cube& c2);

mlpack::FFN<> loadModel(std::string filename);

int saveModel(mlpack::FFN<> model, std::string name);

//#include "Utils.tpp"