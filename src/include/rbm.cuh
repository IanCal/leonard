/*
 *   This file is part of Leonard.
 *
 *   Foobar is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Foobar is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Leonard.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RBM_HEADER
#define RBM_HEADER

class ParameterController;
class InputSource;

/** This is a full, multi-layered RBM
 *
 */
class RBM{

	public:
	
		RBM(int numLayers, int *sizeOfLayers, int *sizeOfLabels, ParameterController *parameterController, InputSource *inputSource, int batchSize);


		/** 
		 * This is used to update the parameters like the learning rates
		 * momentum, decay, etc. ParameterUpdater is an abstract class, there are
		 * certain functions which must be called. Other than that, go
		 * wild.
		 */
		ParameterController *parameterUpdater;
		/**
		 * This controls the input images. It must take care of all memory related to
		 * reading in the images. 
		 */ 
		InputSource *inputSource;
		//! This is the number of images to pass every iteration.
		int batchSize;
		//! The number of contrastive divergence samples to be taken. Default is 1.
		int CDSamples;

		int numberOfNeuronLayers;
		int *layerSizes;
		
		// This is the number of layers of weights
		// It's equal to numberOfNeuronLayers-1
		int numberOfWeightLayers;

		int *labelSizes;
		
		float *learningRates;
		float *biasLearningRates;
		float *weightDecay;
		float *momentum;

		/* 
		 * This is a temporary array, purely for storing things
		 * which need to be transferred to the GPU. It will hold
		 * things like random numbers.
		 */ 
		float *scratch;
		float *d_randomNumbers;
		int amountOfRandomNumbers;
		struct Rand48 *rng;

		// Device pointers
		
		/* 
		 * Array of inputs to each layer (Most will be equal
		 * to an output of another layer, just a nicer way of
		 * referencing them)
		 */
		float **d_input_t0;
		float **d_input_pt0;
		float **d_labels_input;
		// The inputs at time N and probabilities at time N
		// respectively
		float **d_input_tn;
		float **d_input_ptn;
		float **d_labels_reconstruction;
		
		/* 
		 * Array of outputs from each layer (Most will be equal
		 * to an input of another layer, just a nicer way of
		 * referencing them)
		 */
		float **d_output_t0;
		float **d_output_pt0;
		// The inputs at time N and probabilities at time N
		// respectively
		float **d_output_tn;
		float **d_output_ptn;

		// Biases
		
		float **d_inputBiases;
		float **d_outputBiases;

		// Weights
		// Stored in the following way:
		//  To Be Declared
		float **d_weights;

		// 
		


		// Function Declarations

		void alternatingGibbsSampling(int layer, int iterations, bool probabilisticInput=false, bool probabilisticOutput=false, bool startAtTop=false);
		void pushUp(int layer, bool input_t0, bool output_t0, bool useProbabilities);
		void pushDown (int layer, bool input_t0, bool output_t0, bool useProbabilities);

		void updateWeightsInLayer(int layer);
		void updateWeights();
		void setInputPattern();
		void setLabels();
		void getReconstruction(int layer, float *output);
		void getLabels(int layer, float *output);
		void learningIteration();
		void updateBiasesInLayer(int layer);
		void generateRandomNumbers(float scale);
		void setValue(float *device_array, int size, float value=0.f);
		void setRandom(float *device_array, int size, float scale);
};	
#endif

