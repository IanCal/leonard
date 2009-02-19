#include "ParameterController.cuh"

class SimpleController: public ParameterController{
	public:
		float learningRate;
		int samplesPerEpoch;
		int trainingEpochs;
		int currentSample;
		int layerToTrain;
		SimpleController(float iterationLearningRate, int samples, int epochs);
		void updateParameters(RBM *currentRBM);
		void initialise(RBM *currentRBM);
};

