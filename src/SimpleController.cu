#include "SimpleController.cuh"


SimpleController::SimpleController(float iterationLearningRate, int samples, int epochs){
	learningRate = iterationLearningRate;
	samplesPerEpoch = samples;
	trainingEpochs = epochs;
	layerToTrain=0;
	currentSample=0;
};

void SimpleController::updateParameters(RBM *currentRBM){
	
	if( currentSample<samplesPerEpoch*trainingEpochs )
	{
		layerToTrain++;
		for( int layer=0 ; layer<currentRBM->numberOfWeightLayers ; layer++ )
		{
			currentRBM->learningRates[layer]=0.;	
		}
		currentRBM->learningRates[layerToTrain]=learningRate;
	}
	
};

void SimpleController::initialise(RBM *currentRBM){
	for( int layer=0 ; layer<currentRBM->numberOfWeightLayers ; layer++ )
	{
		currentRBM->learningRates[layer]=0.;	
		currentRBM->momentum[layer]=0.;	
	}
	currentRBM->learningRates[0]=learningRate;
};
