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
	d_input_t0 = new float*[numberOfNeuronLayers];
	d_input_tn = new float*[numberOfNeuronLayers];
	d_input_pt0 = new float*[numberOfNeuronLayers];
	d_input_ptn = new float*[numberOfNeuronLayers];
	d_output_t0 = new float*[numberOfNeuronLayers];
	d_output_tn = new float*[numberOfNeuronLayers];
	d_output_pt0 = new float*[numberOfNeuronLayers];
	d_output_ptn = new float*[numberOfNeuronLayers];
	//Biases
	d_inputBiases = (float**)malloc(numberOfNeuronLayers * sizeof(float*));
	d_outputBiases = (float**)malloc(numberOfNeuronLayers * sizeof(float*));

	d_weights = new float*[numberOfWeightLayers];
	
	amountOfRandomNumbers=0;
	int numWBlocks;
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
		setValue(d_inputBiases[layer],layerSizes[layer]+labelSizes[layer]);

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
			checkError(status);
			setValue(d_outputBiases[layer],layerSizes[layer+1]);
			checkError(status);
		}
		if (layerSizes[layer]+labelSizes[layer]>amountOfRandomNumbers)
			amountOfRandomNumbers=layerSizes[layer]+labelSizes[layer];
	}

	amountOfRandomNumbers*=batchSize;
	scratch=new float[amountOfRandomNumbers];

	status |= cublasAlloc(amountOfRandomNumbers, sizeof(float), (void**)&d_randomNumbers);
	checkError(status);
	rng = new Rand48();
	int numBlocks=amountOfRandomNumbers/blockSize + (amountOfRandomNumbers%blockSize == 0?0:1);
	rng->init(numBlocks*blockSize, 123456);

	generateRandomNumbers(1.0);
	for( int layer=0 ; layer<numberOfWeightLayers ; layer++ )
	{
		printf("Setting layer %d\n",layer);
			setRandom(d_weights[layer],(layerSizes[layer]+labelSizes[layer])*layerSizes[layer+1],1.0);
	}

};
void RBM::setValue(float *device_array, int size, float value){
	int numBlocks=size/blockSize + (size%blockSize == 0?0:1);
	setToValue<<<numBlocks,blockSize>>>(device_array,size,value);
	cudaThreadSynchronize();
};
void RBM::setRandom(float *device_array, int size, float scale){

	int numBlocks;
    int nextChunkSize;	
	int currentPosition=0;
	while (currentPosition<size){
		if (size-currentPosition>amountOfRandomNumbers)
			nextChunkSize=amountOfRandomNumbers;
		else
			nextChunkSize=size-currentPosition;
		numBlocks=nextChunkSize/blockSize + (nextChunkSize%blockSize == 0?0:1);
		setRandomScale<<<numBlocks,blockSize>>>(&device_array[currentPosition],d_randomNumbers,nextChunkSize,scale);
		generateRandomNumbers(1.0);
		currentPosition+=nextChunkSize;
	}

};
void RBM::generateRandomNumbers(float scale){
	//int numBlocks=amountOfRandomNumbers/blockSize + (amountOfRandomNumbers%blockSize == 0?0:1);
	//setRand<<<numBlocks,blockSize>>>(d_randomNumbers,amountOfRandomNumbers,*rng);
	for( int i=0 ; i<amountOfRandomNumbers ; i++ )
	{
		scratch[i]=drand48()*scale;
	}
    cublasSetVector(amountOfRandomNumbers, sizeof(float), scratch, 1, d_randomNumbers, 1);
	
	cudaThreadSynchronize();
};

void RBM::pushDown(int layer, bool input_t0, bool output_t0, bool useProbabilities){

	//Basic variables
	
	int inputs = layerSizes[layer]+labelSizes[layer];
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
	checkError(cublasGetError());
	
	//probabilities kernel
	probabilities<<<numberOfBlocks,blockSize>>>(d_input_p, d_inputBiases[layer], inputBatchSize);
	checkError(cublasGetError());
	
	//cutoff kernel
	cutoff<<<numberOfBlocks,blockSize>>>(d_input_p, d_input, d_randomNumbers, inputBatchSize);
	checkError(cublasGetError());

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
	checkError(cublasGetError());
	
	//probabilities kernel
	probabilities<<<numberOfBlocks,blockSize>>>(d_output_p, d_outputBiases[layer], outputBatchSize);
	cudaThreadSynchronize();
	checkError(cublasGetError());
	
	//cutoff kernel
	cutoff<<<numberOfBlocks,blockSize>>>(d_output_p, d_output, d_randomNumbers, outputBatchSize);
	cudaThreadSynchronize();
	checkError(cublasGetError());

};

void RBM::alternatingGibbsSampling(int layer, int iterations, bool probabilisticInput, bool probabilisticOutput, bool startAtTop){

	// Push up the initial pattern, then down to the inputs
	if (!startAtTop)
		pushUp(layer, true, true, true);
	pushDown(layer, false, true, probabilisticOutput);
	//Cycle doing this
	for( int i=0 ; i<iterations-1 ; i++ )
	{
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
			1.0f,d_weights[layer],outputs);
	checkError(cublasGetError());

};

void RBM::updateWeights(){

	int topRequiredLayer=0;

	// We need to know the top layer with a learning rate
	// That way we only push the data up as far as it needs to go
	for( int layer=0 ; layer<numberOfWeightLayers ; layer++ )
	{
		if( learningRates[layer]!=0.0 )
		{
			topRequiredLayer=layer;
		}
	}

	// Now it's time to actually process the data
	for( int layer=0 ; layer<=topRequiredLayer; layer++ ){
		// If no learning rate, no reason to train it
		if( learningRates[layer]==0 ){
			pushUp(layer, true, true, true);
		}
		else{
			// Sample and then update weights
			alternatingGibbsSampling(layer, CDSamples);
			updateWeightsInLayer(layer);
			updateBiasesInLayer(layer);
		}
	}

};

void RBM::setInputPattern(){
	// Set the input 
    cublasSetVector(layerSizes[0]*batchSize, sizeof(float), (inputSource->getNextInput(this)), 1, d_input_pt0[0], 1);
	checkError(cublasGetError());
};

void RBM::getReconstruction(int layer, float *output){
	cublasGetVector(layerSizes[layer]*batchSize, sizeof(float), d_input_ptn[layer], 1, output, 1);
	//cublasGetVector(layerSizes[layer]*batchSize, sizeof(float), d_weights[layer], 1, output, 1);

};

void RBM::learningIteration(){
	setInputPattern();
	updateWeights();
	parameterUpdater->updateParameters(this);
	generateRandomNumbers(1.0f);
};
