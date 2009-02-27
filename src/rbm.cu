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

// Need to cut down on the includes, these are the max required, not sure
// which are needed
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include  <sys/timeb.h>
#include <allegro.h>

//mmap stuff
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>


/* Includes, cuda */
#include "cublas.h"
#include "kernels.cu"
#include "include/rbm.cuh"
#include "include/ParameterController.h"
#include "include/InputSource.h"

void checkError(cublasStatus status, char *message = NULL){
	
	if (status != CUBLAS_STATUS_SUCCESS) {
		if (message!=NULL)
			printf("Error message: %s\n",message);
       if( status == CUBLAS_STATUS_ALLOC_FAILED )
	   {
		  fprintf (stderr, "Error allocating resources on device\n");
	   }
       if( status == CUBLAS_STATUS_INVALID_VALUE )
	   {
		  fprintf (stderr, "Error - invalid value passed\n");
	   } 
       if( status == CUBLAS_STATUS_MAPPING_ERROR )
	   {
		  fprintf (stderr, "Error - Access to memory space failed\n");
	   } 
       if( status == CUBLAS_STATUS_EXECUTION_FAILED )
	   {
		  fprintf (stderr, "Error  - Program failed to execute on device\n");
	   } 
       if( status == CUBLAS_STATUS_INTERNAL_ERROR )
	   {
		  fprintf (stderr, "Error - Internal error in CUBLAS :( \n");
	   } 
		fprintf (stderr, "Exiting...\n");
        exit(1);
	}

};

/* 
 * The actual code for the RBM goes in here
 */

RBM::RBM(int numLayers, int *sizeOfLayers, int *sizeOfLabels, ParameterController *parameterController, InputSource *inputSource, int batchSize){
	printf("Creating RBM with \n");
	CDSamples=1;
	this->batchSize=batchSize;
	numberOfWeightLayers=numLayers-1;
	numberOfNeuronLayers=numLayers;
	layerSizes=sizeOfLayers;
	labelSizes=sizeOfLabels;
	learningRates = new float[numLayers]; 
	momentum = new float[numLayers];
	parameterUpdater = parameterController; 	
	parameterUpdater->initialise(this);
	this->inputSource=inputSource;
	inputSource->initialise(this);
	this->batchSize=batchSize;
	cublasStatus status;
	status =  CUBLAS_STATUS_SUCCESS;

	// Build up the device arrays

	d_input_t0 = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_input_pt0 = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_input_tn = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_input_ptn = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_output_t0 = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_output_pt0 = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_output_tn = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_output_ptn = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	//Biases
	d_inputBiases = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_outputBiases = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	//Weights
	d_weights = (float**)malloc(numberOfWeightLayers * sizeof(float*));
	
	amountOfRandomNumbers=0;
	for( int layer=0 ; layer<numberOfNeuronLayers ; layer++ )
	{
		//set 
		status |= cublasAlloc((labelSizes[layer]+layerSizes[layer])*batchSize, sizeof(float), (void**)&d_input_t0[layer]);
		checkError(status);
		status |= cublasAlloc((labelSizes[layer]+layerSizes[layer])*batchSize, sizeof(float), (void**)&d_input_pt0[layer]);
		checkError(status);
		status |= cublasAlloc((labelSizes[layer]+layerSizes[layer])*batchSize, sizeof(float), (void**)&d_input_tn[layer]);
		checkError(status);
		status |= cublasAlloc((labelSizes[layer]+layerSizes[layer])*batchSize, sizeof(float), (void**)&d_input_ptn[layer]);
		status |= cublasAlloc(layerSizes[layer]+labelSizes[layer], sizeof(float), (void**)&d_inputBiases[layer]);
		checkError(status);
		setZero(d_inputBiases[layer],layerSizes[layer]+labelSizes[layer]);

		// now the output side of things
		if( layer>0 )
		{
			d_output_t0[layer-1]=d_input_t0[layer];
			d_output_pt0[layer-1]=d_input_pt0[layer];
			d_output_tn[layer-1]=d_input_tn[layer];
			d_output_ptn[layer-1]=d_input_ptn[layer];
		}
		if (layer<numberOfWeightLayers)
		{
			status |= cublasAlloc((layerSizes[layer]+labelSizes[layer])*layerSizes[layer+1], sizeof(float), (void**)&d_weights[layer]);
		checkError(status);
			status |= cublasAlloc(layerSizes[layer+1], sizeof(float), (void**)&d_outputBiases[layer]);
		checkError(status);
			setZero(d_weights[layer],(layerSizes[layer]+labelSizes[layer])*layerSizes[layer+1]);
		checkError(status);
			setZero(d_outputBiases[layer],layerSizes[layer+1]);
		checkError(status);
			if ((layerSizes[layer]+labelSizes[layer])*layerSizes[layer+1]>amountOfRandomNumbers)
				amountOfRandomNumbers=(layerSizes[layer]+labelSizes[layer])*layerSizes[layer+1];
		}
	}


	scratch = (float*)malloc(amountOfRandomNumbers * sizeof(float));
	status |= cublasAlloc(amountOfRandomNumbers, sizeof(float), (void**)&d_randomNumbers);
		checkError(status);
	generateRandomNumbers(1.0);
	
};
void RBM::setZero(float *device_array, int size){
	int numBlocks=size/blockSize + (size%blockSize == 0?0:1);
	setToValue<<<numBlocks,blockSize>>>(device_array,size,0);
	cudaThreadSynchronize();
};
void RBM::generateRandomNumbers(float scale){
	for( int i=0 ; i<amountOfRandomNumbers ; i++ )
	{
		scratch[i]=drand48();
	}
	cublasSetVector(amountOfRandomNumbers, sizeof(float), scratch, 1, d_randomNumbers, 1);
	cudaThreadSynchronize();
};

void RBM::pushDown(int layer, bool input_t0, bool output_t0, bool useProbabilities){

	//Basic variables
	
	int inputs = layerSizes[layer];
	int outputs = layerSizes[layer+1];
	int inputBatchSize = inputs * batchSize;
	int numberOfBlocks = inputBatchSize/blockSize + (inputBatchSize%blockSize == 0?0:1);

	//device pointers
	
	float* d_input;
	float* d_input_p;
	float* d_output;
	if (input_t0){
		d_input_p = d_input_pt0[layer];
		d_input = d_input_t0[layer];	
	}
	else{
		d_input_p = d_input_ptn[layer];
		d_input = d_input_tn[layer];
	}

	if (output_t0){
		if (useProbabilities)
			d_output = d_output_pt0[layer];
		else
			d_output = d_output_t0[layer];
	}
	else{
		if (useProbabilities)
			d_output = d_output_ptn[layer];
		else
			d_output = d_output_tn[layer];
	}
	//Matrix multiplication
	cudaThreadSynchronize();
	cublasSgemm('n','n',batchSize,inputs,outputs,
			1.f,d_output,batchSize,
			d_weights[layer],outputs,
			0.f,d_input_p,batchSize);
	
	//probabilities kernel
	probabilities<<<numberOfBlocks,blockSize>>>(d_input_p, d_inputBiases[layer], inputBatchSize);
	
	//cutoff kernel
	cutoff<<<numberOfBlocks,blockSize>>>(d_input_p, d_input, d_randomNumbers, inputBatchSize);

};

void RBM::pushUp(int layer, bool input_t0, bool output_t0, bool useProbabilities){

	//Basic variables
	
	int inputs = layerSizes[layer]+labelSizes[layer];
	int outputs = layerSizes[layer+1];
	int outputBatchSize = outputs * batchSize;
	int numberOfBlocks = outputBatchSize/blockSize + (outputBatchSize%blockSize == 0?0:1);

	//device pointers
	float* d_input;
	float* d_output_p;
	float* d_output;
	if (input_t0){
		if (useProbabilities)
			d_input = d_input_pt0[layer];
		else
			d_input = d_input_t0[layer];	
	}
	else{
		if (useProbabilities)
			d_input = d_input_ptn[layer];
		else
			d_input = d_input_tn[layer];	
	}
	
	if (output_t0){
		d_output_p = d_output_pt0[layer];
		d_output = d_output_t0[layer];
	}
	else{
		d_output_p = d_output_ptn[layer];
		d_output = d_output_tn[layer];
	}
	//Matrix multiplication
	cudaThreadSynchronize();
	cublasSgemm('n','T',batchSize,outputs,inputs,
			1.f,d_input,batchSize,
			d_weights[layer],outputs,
			0.f,d_output_p,batchSize);
	
	//probabilities kernel
	probabilities<<<numberOfBlocks,blockSize>>>(d_output_p, d_outputBiases[layer], outputBatchSize);
	cudaThreadSynchronize();
	
	//cutoff kernel
	cutoff<<<numberOfBlocks,blockSize>>>(d_output_p, d_output, d_randomNumbers, outputBatchSize);
	cudaThreadSynchronize();

};

void RBM::alternatingGibbsSampling(int layer, int iterations, bool probabilisticInput, bool probabilisticOutput, bool startAtTop){

	// Push up the initial pattern, then down to the inputs
	if (!startAtTop)
		pushUp(layer, true, true, probabilisticInput);
	pushDown(layer, false, true, probabilisticOutput);
	//Cycle doing this
	for( int i=0 ; i<iterations-1 ; i++ )
	{
		printf("Gibbs!!\n");
		pushUp(layer, false, false, probabilisticInput);
		pushDown(layer, false, false, probabilisticOutput);
	}
	//Final push up.	
	pushUp(layer, false, false, probabilisticInput);
	
};

void RBM::updateBiasesInLayer(int layer){
	int inputs = layerSizes[layer]+labelSizes[layer];
	int outputs = layerSizes[layer+1];
	int inputBatchSize = inputs * batchSize;
	int outputBatchSize = outputs * batchSize;

	int nBlocksForInBiases = inputs/blockSize + (inputs%blockSize == 0?0:1);
	int nBlocksForOutBiases = outputs/blockSize + (outputs%blockSize == 0?0:1);

	// Update the input biases
	biasesIncrease<<<nBlocksForInBiases,blockSize>>>(d_input_pt0[layer], d_inputBiases[layer], biasLearningRates[layer], inputBatchSize/batchSize);
	biasesDecrease<<<nBlocksForInBiases,blockSize>>>(d_input_ptn[layer], d_inputBiases[layer], biasLearningRates[layer], inputBatchSize/batchSize, 0.0);

	// Update the output biases
	biasesIncrease<<<nBlocksForOutBiases,blockSize>>>(d_output_pt0[layer], d_outputBiases[layer], biasLearningRates[layer], outputBatchSize/batchSize);
	biasesDecrease<<<nBlocksForOutBiases,blockSize>>>(d_output_ptn[layer], d_outputBiases[layer], biasLearningRates[layer], outputBatchSize/batchSize, 0.0);

};

void RBM::updateWeightsInLayer(int layer){
	int inputs = layerSizes[layer]+labelSizes[layer];
	int outputs = layerSizes[layer+1];
	// Update the weights
	cublasSgemm('T','n',outputs,inputs,batchSize,
			learningRates[layer],d_output_pt0[layer],batchSize,
			d_input_pt0[layer],batchSize,
			1.f,d_weights[layer],outputs);
	checkError(cublasGetError());

	cublasSgemm('T','n',outputs,inputs,batchSize,
			-learningRates[layer],d_output_ptn[layer],batchSize,
			d_input_ptn[layer],batchSize,
			1.f,d_weights[layer],outputs);
	checkError(cublasGetError());

};

void RBM::updateWeights(){

	int topRequiredLayer=0;

	// We need to know the top layer with a learning rate
	for( int layer=0 ; layer<numberOfWeightLayers ; layer++ )
	{
		if( learningRates[layer]!=0.0 )
		{
			topRequiredLayer=layer;
		}
	}

	for( int layer=0 ; layer<=topRequiredLayer; layer++ ){
		if( learningRates[layer]==0 ){
			pushUp(layer, true, true, true);
		}
		else{
			alternatingGibbsSampling(layer, CDSamples);
			updateWeightsInLayer(layer);
		}
	}

};

void RBM::setInputPattern(){
  cublasSetVector(layerSizes[0]*batchSize, sizeof(float), (inputSource->getNextInput(this)), 1, d_input_pt0[0], 1);
//	cublasSetVector(layerSizes[0]*batchSize, sizeof(float), scratch, 1, d_input_pt0[0], 1);
	cudaThreadSynchronize();
	checkError(cublasGetError());
};

void RBM::learningIteration(){
	
	setInputPattern();
	updateWeights();
	parameterUpdater->updateParameters(this);
};
