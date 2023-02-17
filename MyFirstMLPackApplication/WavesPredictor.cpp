#include "WavesPredictor.h"
#include "Utils.h"
#include <typeinfo>
#include "AsciiGraphicsPrintLoss.h"

#define INPUT_SEQ_LEN 32
#define NEURONS_COUNT 128
#define LAYERS_COUNT 2//слоёв немного: обычно 3-4, 7-8 уже тяжело
#define DROPOUT 0.0
#define EPOCHS_COUNT 100
#define LEARNING_RATE 0.0005
//rule of thumb with regard to batch size is that it should be as big as memory permits 
// but at most 1% of the number of observations.
#define BATCH_SIZE 32
#define MIN_PATIENCE 5
#define MAX_PATIENCE 10

using namespace mlpack;
using namespace ens;


//std::tuple<RNN<MeanSquaredError, HeInitialization>, AdamType<>> createModel(size_t inputSize, size_t outputSize, 
RNN<MeanSquaredError, HeInitialization> createModel(std::ostream& o, size_t inputSize, size_t outputSize, AdamType<> optimizer,
    int epochsCount, int neuronsCount, int additionalLayersCount, double dropout, arma::cube &trainCube, arma::cube &trainLabelsCube) {

    // Так, функция активации ReLU даёт хорошие результаты точности в полносвязных слоях(Dense layers) и свёрточных слоях(Convolutional layers).
     // А функция гиперболического тангенса Tahn наравне с функцией Sigmoid используются в рамках моделей LSTM(Long Short Term Memory).
     //RNN<MeanSquaredError, HeInitialization> model(seqLen, true /* one response per sequence */);
    RNN<MeanSquaredError, HeInitialization> model(inputSize, false /* few responses per sequence */);

    // Dropout и BatchNorm (Layer Norm) между слоями
    // LSTM -> Dropout -> Dense -> Softmax (вариант из лекций)
    // LSTM -> Dropout -> LSTM -> Dropout -> Dense -> Softmax (вариант из лекций)
    // Number of cells in the LSTM (hidden layers in standard terms).
    int layerSize = neuronsCount / (1 + additionalLayersCount);
    for (int i = 0; i < additionalLayersCount; i++) {
        /*model.Add<LSTM>(H1);
        model.Add<LinearNoBias>(predLen);*/
        model.Add<LSTM>(layerSize);
        if (dropout > 0)
            model.Add<Dropout>(dropout);
        //precision dramatically increased since I removed LeakyReLu after inner layers
        //model.Add<LeakyReLU>();
        //model.Add<Sigmoid>();
        //model.Add<TanH>();
    }

    model.Add<LSTM>(layerSize);
    model.Add<LeakyReLU>();
    model.Add<Linear>(outputSize);

    o << "Model created: " << additionalLayersCount + 1 << " layers, " << (additionalLayersCount + 1) * layerSize 
        << " neurons, " << dropout << " dropout." << std::endl;
    print_layers(o, model);

    model.Train(trainCube, trainLabelsCube, optimizer,
        // PrintLoss Callback prints loss for each epoch.
        AsciiGraphicsPrintLoss(o, fminl(10, EPOCHS_COUNT / 10)),
        // Progressbar Callback prints progress bar for each epoch.
        //ens::ProgressBar(),
        // Stops the optimization process if the loss stops decreasing or no improvement has been made.
        // This will terminate the optimization once we obtain a minima on training set.
        ens::EarlyStopAtMinLoss(fmaxl(MIN_PATIENCE, fminl(epochsCount / 5, MAX_PATIENCE))),
        // Report Callback prints final report at the end of he training
        ens::Report(0.1)
    );

    //return std::make_tuple(model, optimizer);
    return model;
}

template<typename OptimizerType>
OptimizerType createOptimizer(std::ostream& o, int pointsCount, int epochsCount) {
    //TODO pass as arguments
    //int tolerance = -1;
    int tolerance = 1e-8;
    double stepSize = LEARNING_RATE;
    size_t batchSize = BATCH_SIZE;
    double meanSquareGradParamInit = 0.00000001;

    //each epoch will be = number of points / batchSize
    size_t maxPointIterations = epochsCount * pointsCount;

    //StandardSGD optimizer(stepSize, batchSize, maxPointIterations, tolerance);
    ens::AdamType<ens::AdamUpdate> optimizer(stepSize, batchSize, 0.9, 0.999, meanSquareGradParamInit, maxPointIterations, tolerance);
    //ens::AdamType<ens::RMSPropUpdate> optimizer(stepSize, batchSize, 0.9, 0.999, meanSquareGradParamInit, maxPointIterations, tolerance);
    //ens::RMSProp optimizer(stepSize, batchSize, 0.99, meanSquareGradParamInit, maxPointIterations, tolerance);
    o << "Optimizer params: " << epochsCount << " epochs, " << stepSize << " learning rate, " << batchSize << " batchsize." << std::endl;
    return optimizer;
}

//template AdamType<> createOptimizer<AdamType<>>(std::ostream& o, int pointsCount);


int WavesPredictor::simplePredict() {
    int seqLen = INPUT_SEQ_LEN;
    const double dropout = DROPOUT;
    int layersCount = LAYERS_COUNT;
    int neuronsCount = NEURONS_COUNT;
    int epochsCount = EPOCHS_COUNT;
    int predValuesWidth = 2;
    int analSeqWidth = 3;

    int slicesPrintCount = 3;
    float slicesPrintRatio = 0.4f;

    auto sets = loadDataSetAlt("WAVESUSDT_19.09.01-23.02.07_4H_export.txt", 0.75f, seqLen);
    arma::mat trainData = std::get<0>(sets).rows(1, 3);
    arma::mat testData = std::get<1>(sets).rows(1, 3);
    std::cout << "Train data loaded:" << std::endl;
    normal_print(std::cout, trainData, true);
    std::cout << "Test data loaded:" << std::endl;
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

    std::cout << "Data normalized:" << std::endl;
    normal_print(std::cout, trainData, true);
    normal_print(std::cout, testData, true);
    std::cout << std::endl;

    auto trainCubes = preparePredictionCubes(trainData, analSeqWidth, seqLen, predValuesWidth);
    arma::cube trainCube = std::get<0>(trainCubes);
    arma::cube trainLabelsCube = std::get<1>(trainCubes);
    /*std::cout << "Train data prepared:" << std::endl;
    normal_print(std::cout, trainCube, 2, 0.5f, true);*/

    //trainCube.brief_print(std::cout, "Train data");
   /* std::cout << "Train data (first and last slices)" << std::endl;
    normal_print(std::cout, trainCube.slice(0), true);
    normal_print(std::cout, trainCube.slice(trainCube.n_slices - 1), true);
    //trainLabelsCube.brief_print(std::cout, "Train labels");
    std::cout << "Train labels (first and last slices)" << std::endl;
    normal_print(std::cout, trainLabelsCube.slice(0), true);
    normal_print(std::cout, trainLabelsCube.slice(trainLabelsCube.n_slices - 1), true);
    std::cout << std::endl;*/

    
    auto optimizer = createOptimizer<AdamType<>>(std::cout, trainCube.n_cols, epochsCount);
    RNN<MeanSquaredError, HeInitialization> model = createModel(std::cout, seqLen, predValuesWidth, optimizer, epochsCount,
        neuronsCount, layersCount - 1, dropout, trainCube, trainLabelsCube);

    //Testing model on training data
    arma::cube trainPredictions;
    model.Predict(trainCube, trainPredictions);//2 rows, 5647 cols, <seqLen> slices
    std::cout << std::endl;

    //Denormalizinig predictions
    for (int i = 0; i < trainPredictions.n_rows; i++) {
        trainPredictions.row(i) *= max[i];
    }

    for (int i = 0; i < trainLabelsCube.n_rows; i++) {
        trainLabelsCube.row(i) *= max[i];
    }

    std::cout << std::endl;
    std::cout << "Real training labels:" << std::endl;
    normal_print(std::cout, trainLabelsCube, slicesPrintCount, slicesPrintRatio, true);
    std::cout << std::endl;
    std::cout << "Predicted labels:" << std::endl;
    normal_print(std::cout, trainPredictions, slicesPrintCount, slicesPrintRatio, true);

    // Calculate MSE on prediction.
    double trainMSEP = ComputeMSE(trainPredictions, trainLabelsCube);
    std::cout << "Mean Squared Error on prediction data points in training set := " << trainMSEP << std::endl;

    // Calculate percental error
    std::array<double, 3> percentsTrain = calculateDifferencePrecents(trainLabelsCube, trainPredictions);
    std::cout << "Deviation in percents (min, avg, max): " << percentsTrain[0] << " " << percentsTrain[1] << " " << percentsTrain[2] << std::endl;

    //Testing model on test data
    auto testCubes = preparePredictionCubes(testData, analSeqWidth, seqLen, predValuesWidth);
    arma::cube testCube = std::get<0>(testCubes);
    arma::cube testLabelsCube = std::get<1>(testCubes);
    /*std::cout << "Test data prepared:" << std::endl;
    normal_print(std::cout, testCube, 2, 0.5f, true);*/

    arma::cube testPredictions;
    model.Predict(testCube, testPredictions);//2 rows, 5647 cols, <seqLen> slices

    //Denormalizinig predictions
    for (int i = 0; i < trainPredictions.n_rows; i++) {
        testPredictions.row(i) *= max[i];
    }

    for (int i = 0; i < trainLabelsCube.n_rows; i++) {
        testLabelsCube.row(i) *= max[i];
    }

    std::cout << "Real test labels:" << std::endl;
    normal_print(std::cout, testLabelsCube, slicesPrintCount, slicesPrintRatio, true);
    std::cout << "Predicted labels:" << std::endl;
    normal_print(std::cout, testPredictions, slicesPrintCount, slicesPrintRatio, true);

    // Calculate MSE on prediction.
    double testMSEP = ComputeMSE(testPredictions, testLabelsCube);
    std::cout << "Mean Squared Error on prediction data points in test set := " << testMSEP << std::endl;

    // Calculate percental error
    std::array<double, 3> percentsTest = calculateDifferencePrecents(testLabelsCube, testPredictions);
    std::cout << "Deviation in percents (min, avg, max): " << percentsTest[0] << " " << percentsTest[1] << " " << percentsTest[2] << std::endl;

    return 0;
}