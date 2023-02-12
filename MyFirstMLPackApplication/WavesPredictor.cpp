#include "WavesPredictor.h"
#include "Utils.h"

using namespace mlpack;
using namespace ens;

int WavesPredictor::simplePredict() {
    int seqLen = 99;
    int predLen = 2;
    int analSeqCount = 3;
    // Number of cells in the LSTM (hidden layers in standard terms).
    // NOTE: you may play with this variable in order to further optimize the
    // model (as more cells are added, accuracy is likely to go up, but training
    // time may take longer).
    const int H1 = 25;

    auto sets = loadDataSetAlt("WAVESUSDT_19.09.01-23.02.07_4H_export.txt", 0.75f, seqLen);
    arma::mat trainData = std::get<0>(sets).rows(1, 3);
    arma::mat testData = std::get<1>(sets).rows(1, 3);
    std::cout << "Train data loaded:" << std::endl;
    normal_print(std::cout, trainData, true);
    normal_print(std::cout, testData, true);
    //loaded fine
    std::cout << std::endl;

    // Scale all data into the range (0, 1) for increased numerical stability.
    /*data::MinMaxScaler scale;
    // Fit scaler only on training data.
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);*/
    //float max = std::fmaxf(trainData.max(), testData.max());

    std::array<float, 3> max;
    for (int i = 0; i < trainData.n_rows; i++) {
        max[i] = trainData.row(i).max();
        trainData.row(i) /= max[i];
        testData.row(i) /= max[i];
    }

    std::cout << "Train data normalized:" << std::endl;
    normal_print(std::cout, trainData, true);
    normal_print(std::cout, testData, true);

    auto trainCubes = preparePredictionCubes(trainData, analSeqCount, seqLen, predLen);
    arma::cube trainCube = std::get<0>(trainCubes);
    arma::cube trainLabelsCube = std::get<1>(trainCubes);
    std::cout << "Train data prepared:" << std::endl;
    //trainCube.brief_print(std::cout, "Train data");
    std::cout << "Train data (first and last slices)" << std::endl;
    normal_print(std::cout, trainCube.slice(0), true);
    normal_print(std::cout, trainCube.slice(trainCube.n_slices - 1), true);
    //trainLabelsCube.brief_print(std::cout, "Train labels");
    std::cout << "Train labels (first and last slices)" << std::endl;
    normal_print(std::cout, trainLabelsCube.slice(0), true);
    normal_print(std::cout, trainLabelsCube.slice(trainLabelsCube.n_slices - 1), true);

    std::cout << std::endl;
    //RNN<MeanSquaredError, HeInitialization> model(seqLen, true /* one response per sequence */);
    RNN<MeanSquaredError, HeInitialization> model(seqLen, false /* few responses per sequence */);
    /*model.Add<LSTM>(H1);
    model.Add<LinearNoBias>(predLen);*/
    model.Add<LSTM>(H1);
    model.Add<Dropout>(0.5);
    model.Add<LeakyReLU>();

    model.Add<LSTM>(H1);
    model.Add<Dropout>(0.5);
    model.Add<LeakyReLU>();

    model.Add<LSTM>(H1);
    model.Add<LeakyReLU>();
    model.Add<Linear>(predLen);

    std::cout << model.NumFunctions() << std::endl;

    int epochs = 25;
    int tolerance = -1;
    double stepSize = 0.0001;
    //double stepSize = 0.001;
    size_t batchSize = 32;//rule of thumb with regard to batch size is that it should be as big as memory permits but at most 1% of the number of observations.
    size_t maxPointIterations = epochs * trainCube.n_cols;
    double meanSquareGradParamInit = 0.00000001;
    //each epoch will be = number of points / batchSize

    //StandardSGD optimizer(stepSize, batchSize, maxPointIterations, tolerance);
    ens::AdamType<ens::AdamUpdate> optimizer(stepSize, batchSize, 0.9, 0.99, meanSquareGradParamInit, maxPointIterations, tolerance);

    model.Train(trainCube, trainLabelsCube, optimizer,
        // PrintLoss Callback prints loss for each epoch.
        ens::PrintLoss(),
        // Progressbar Callback prints progress bar for each epoch.
        ens::ProgressBar(),
        // Stops the optimization process if the loss stops decreasing
        // or no improvement has been made. This will terminate the
        // optimization once we obtain a minima on training set.
        ens::EarlyStopAtMinLoss());

    //TODO check deviation on normalized data, then on denormalized
    //Denormalizinig predictions
    arma::cube trainPredictions;
    model.Predict(trainCube, trainPredictions);//2 rows, 5647 cols, 99 slices
    for (int i = 0; i < trainPredictions.n_rows; i++) {
        trainPredictions.row(i) *= max[i];
    }

    normal_print(std::cout, trainPredictions.slice(0), true);
    normal_print(std::cout, trainPredictions.slice(1), true);
    normal_print(std::cout, trainPredictions.slice(trainPredictions.n_slices - 1), true);
    //trainPredictions.brief_print(std::cout, "");
    //TODO: print percental deviation

    return 0;
}