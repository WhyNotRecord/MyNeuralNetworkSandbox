#pragma once
class WavesPredictor
{
public:
	//std::tuple<RNN<MeanSquaredError, HeInitialization>, AdamType<>> createModel(size_t inputSize, size_t outputSize, int neuronsCount, int additionalLayersCount, double dropout);

	int simplePredict();
};

