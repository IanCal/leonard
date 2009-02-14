/*
 *   This file is part of RBM-on-GPU.
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
 *   along with RBM-on-GPU.  If not, see <http://www.gnu.org/licenses/>.
 */

//#define DISPLAY
#define COLMAJORPOSITION(X, BATCH, BATCHSIZE) ((X*BATCHSIZE) + BATCH)
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include  <sys/timeb.h>

//mmap stuff
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>


/* Includes, cuda */
#include "cublas.h"

#include <allegro.h>
#include "kernels.cu"

#define blockSize (512)
#define WIDTH   800
#define HEIGHT  800
#define BATCHSIZE 32
#define WEIGHTDECAY 1.0


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

//Batchsize=1 is rowMajor equivalent
void drawImage(BITMAP *buffer, int xPos, int yPos, int scale, int width, int height, float *data, 
		int batchSize=1, int displayNumber=0, int spacing=0, int backgroundColour=0){
	
	int xCoord=0;
	int yCoord=0;
	int colour=0;
	if (backgroundColour)
		rectfill(buffer, xPos-spacing, yPos-spacing, xPos+(width*scale)+spacing, yPos+(height*scale)+spacing, makecol(backgroundColour,backgroundColour,backgroundColour));
	for( int i=0 ; i<width ; i++ )
	{
		for( int j=0 ; j<height ; j++ )
		{
			xCoord=xPos+(i*scale);
			yCoord=yPos+(j*scale);
			colour=(int)(data[batchSize*(j*width + i)+displayNumber]*255);
			rectfill(buffer, xCoord+spacing, yCoord+spacing, xCoord+scale-spacing, yCoord+scale-spacing, makecol(colour,colour,colour));
		}
		
	}
	
};	
void drawImageRowMajor(BITMAP *buffer, int xPos, int yPos, int scale, int width, int height, float *data, 
		int batchSize=1, int displayNumber=0, int spacing=0, int backgroundColour=0){
	
	int xCoord=0;
	int yCoord=0;
	int colour=0;
	if (backgroundColour)
		rectfill(buffer, xPos-spacing, yPos-spacing, xPos+(width*scale)+spacing, yPos+(height*scale)+spacing, makecol(backgroundColour,backgroundColour,backgroundColour));
	for( int i=0 ; i<width ; i++ )
	{
		for( int j=0 ; j<height ; j++ )
		{
			xCoord=xPos+(i*scale);
			yCoord=yPos+(j*scale);
			colour=(int)(data[batchSize*displayNumber + (height*j + i)]*125 + 125);
			rectfill(buffer, xCoord+spacing, yCoord+spacing, xCoord+scale-spacing, yCoord+scale-spacing, makecol(colour,colour,colour));
		}
		
	}
	
};	



void convertToColumnMajor(float *in, float *out, int amount, int batchSize){
	assert(amount%batchSize==0);
	for( int batchItem=0 ; batchItem<batchSize ; batchItem++ )
	{
		for( int j=0 ; j<(amount/batchSize) ; j++ )
		{
			out[batchItem+(j*batchSize)]=in[j+(amount/batchSize)*batchItem];
		}
	}
};

float difference(float* a, float* b, int length){
	float diff=0.;
	for( int i=0 ; i<length ; i++ )
	{
		diff+=abs(a[i]-b[i]);
	}
	return diff;	
};


class Layer{
	public:
	/* This will be a single layer of a restricted boltzmann machine
	 *
	 * It will create and allocate all of the device and host memories.
	 * All functions pertaining to the layer itself will go in here,
	 * however it will serve largely as a holder for the arrays.
	 *
	 * There will exist helper functions to call the larger device 
	 * functions (e.g. updateWeights could be simplified as we know
	 * where the arrays are and what they are called. Only the input
	 * pattern needs to be given, the rest are holders.
	 */
	int inputSize;
	int outputSize;

	float* input_t0;
	float* weights;
	float* randomNumbers;

	/*	Weights Format:
	 *	Matrix is in the following format
	 *	
	 *	| 0 | 3 |
	 *	| 1 | 4 |
	 *	| 2 | 5 |
	 *	
	 *	This is very, very important!
	 *	Columns are per input neuron
	 *	Rows are per output neuron
	 *
	 */
	int i;
	//Device arrays
	float* d_input_t0;
	float* d_input_p_t1;
	float* d_input_t1;
	float* d_output_t0;
	float* d_output_p_t0;
	float* d_output_t1;
	float* d_output_p_t1;
	float* d_weights;
	float* d_randomNumbers;
	float* d_labels_in;
	float* d_labels_out;
	float* d_labels_totals;
	float* d_visible_biases;
	float* d_hidden_biases;
	float learningRate;	
	int batchSize;
	float biasLearningRate;
	int numberOfLabels;
	Layer(int numberOfInputs, int numberOfOutputs, int batchsize, float learnRate, float biasLearnRate, int numLabels=0);
	void initializeWeights();
	void generateRandomNumbers();
	void updateWithPattern(float *d_input_pattern);
	void printGenerated(bool quiet);
		
void makeRandom(float *target, int length);
void makeRandomSmall(float *target, int length, float scale);
	/* 
	 * Push the data from the input to the output.
	 * Returns a pointer to the output, which is
	 * also stored in d_output_t0 or d_output_p_t0
	 * depending on the value of cutoff.
	 *
	 * float *d_input_pattern:
	 * 		Input pattern must be on the device memory
	 * 		
	 * bool cutoff:
	 *  	Bad naming, I know. The choice is whether or not you 
	 */
	 float* pushUp(float *d_input_pattern, float *d_output, bool cut_off);
	 float* pushDown(float *d_output, bool cut_off);
	 void setLabels(float* labels, float* position=NULL);
	 void getLabels(float* output);
};

Layer::Layer(int numberOfInputs, int numberOfOutputs, int batchsize, float learnRate, float biasLearnRate, int numLabels){
	numberOfLabels=numLabels;	
	numberOfInputs+=numLabels;	
	learningRate=learnRate;
	biasLearningRate=biasLearnRate;
	batchSize = batchsize;
	inputSize = numberOfInputs*batchSize;
	outputSize = numberOfOutputs*batchSize;
	/* Memory allocation on host
	 *
	 * */

	input_t0 = (float*)malloc(inputSize * sizeof(float));
	weights = (float*)malloc(numberOfInputs*numberOfOutputs * sizeof(float));
	randomNumbers = (float*)malloc( ((outputSize*2)+inputSize) * sizeof(float) );

	/* Memory allocation on device. Really ugly...
	 *
	 * */
    cublasStatus status;
	int memoryUsage=0;
	status =  CUBLAS_STATUS_SUCCESS;
    status |= cublasAlloc(numberOfInputs*numberOfOutputs, sizeof(d_weights[0]), (void**)&d_weights);
	printf("%d\n",numberOfInputs*numberOfOutputs);
	checkError(status,"Allocating weights");
    memoryUsage+=numberOfInputs*numberOfOutputs*sizeof(float);
	status |= cublasAlloc(inputSize, 	sizeof(d_input_t0[0]), 		(void**)&d_input_t0);
	//d_labels_in=d_input_t0+(inputSize-(numLabels*batchSize));//*sizeof(float);
	d_labels_in=&d_input_t0[(numberOfInputs-numLabels)*batchSize];//*sizeof(float);

	checkError(status,"Input vector");
    memoryUsage+=inputSize*sizeof(float);
    status |= cublasAlloc(inputSize, 	sizeof(d_input_p_t1[0]), 	(void**)&d_input_p_t1);
	checkError(status);
    status |= cublasAlloc(BATCHSIZE, 	sizeof(d_input_p_t1[0]), 	(void**)&d_labels_totals);
	checkError(status);
	//d_labels_out=d_input_p_t1+(inputSize-(numLabels*batchSize));//*sizeof(float);
	d_labels_out=&d_input_p_t1[(numberOfInputs-numLabels)*batchSize];//*sizeof(float);
    memoryUsage+=inputSize*sizeof(float);
    status |= cublasAlloc(inputSize, 	sizeof(d_input_t1[0]), 		(void**)&d_input_t1);
	checkError(status);
    memoryUsage+=inputSize*sizeof(float);
    status |= cublasAlloc(outputSize, 	sizeof(d_output_p_t0[0]), 	(void**)&d_output_p_t0);
	checkError(status);
    memoryUsage+=outputSize*sizeof(float);
    status |= cublasAlloc(outputSize, 	sizeof(d_output_t0[0]), 	(void**)&d_output_t0);
	checkError(status);
    memoryUsage+=outputSize*sizeof(float);
    status |= cublasAlloc(outputSize, 	sizeof(d_output_p_t1[0]), 	(void**)&d_output_p_t1);
	checkError(status);
    memoryUsage+=outputSize*sizeof(float);
    status |= cublasAlloc(outputSize, 	sizeof(d_output_t1[0]), 	(void**)&d_output_t1);
	checkError(status);
    memoryUsage+=outputSize*sizeof(float);
	// Work out how many are actually needed here, because it's a known amount
    status |= cublasAlloc(outputSize, sizeof(d_randomNumbers[0]), (void**)&d_randomNumbers);
    memoryUsage+=(outputSize*sizeof(float));	
	checkError(status);
	
    status |= cublasAlloc(numberOfInputs, sizeof(float), (void**)&d_visible_biases);
    memoryUsage+=(numberOfInputs*sizeof(float));	
	checkError(status);
    status |= cublasAlloc(numberOfOutputs, sizeof(float), (void**)&d_hidden_biases);
    memoryUsage+=(numberOfOutputs*sizeof(float));	
	checkError(status);
	printf("Memory usage for this layer is %2.2fMB\n",((float)memoryUsage)/1.e6);

	initializeWeights();
};


float* Layer::pushDown(float *d_output, bool cut_off){

	int random_offset=0;
	// Setup device settings
	int nBlocksForIn = inputSize/blockSize + (inputSize%blockSize == 0?0:1);


	int inputs=inputSize/batchSize;
	int outputs=outputSize/batchSize;


	cudaThreadSynchronize();
	cublasSgemm('n','n',batchSize,inputs,outputs,
			1.f,d_output,batchSize,
			d_weights,outputs,
			0.f,d_input_p_t1,batchSize);
	probabilities<<<nBlocksForIn,blockSize>>>(d_input_p_t1, d_visible_biases, inputSize);
	cudaThreadSynchronize();
	checkError(cublasGetError());

	if( !cut_off )
	{
		return d_input_p_t1;
	}
	else{
		cutoff<<<nBlocksForIn,blockSize>>>(d_input_p_t1,d_input_t1,d_randomNumbers,random_offset,inputSize);
		return d_input_t1;
	};

};

float* Layer::pushUp(float *d_input_pattern, float *d_output, bool cut_off){
	int random_offset=1;
	// Setup device settings
	int nBlocksForOut = outputSize/blockSize + (outputSize%blockSize == 0?0:1);


	int inputs=inputSize/batchSize;
	int outputs=outputSize/batchSize;
	cudaThreadSynchronize();
	cublasSgemm('n','T',batchSize,outputs,inputs,
			1.f,d_input_pattern,batchSize,
			d_weights,outputs,
			0.f,d_output,batchSize);
	probabilities<<<nBlocksForOut,blockSize>>>(d_output, d_hidden_biases, outputSize);
	cudaThreadSynchronize();
	checkError(cublasGetError());
	cudaThreadSynchronize();

	// Forward pass
	//cublasSgemv('n', outputSize, inputSize, 1.f, d_weights, outputSize, d_input_pattern, 1, 0.f, d_output_p_t0, 1);   		//cudaThreadSynchronize();
	if( !cut_off )
	{
		return d_output;
	}
	else{
		cutoff<<<nBlocksForOut,blockSize>>>(d_output,d_output,d_randomNumbers,random_offset,outputSize);
		return d_output;
	};
};
void Layer::generateRandomNumbers(){
	//for (i=0;i<(outputSize*2)+inputSize;i++){
	for (i=0;i<outputSize;i++){
		randomNumbers[i]=drand48();
	};
	cublasSetVector(outputSize, sizeof(weights[0]), randomNumbers, 1, d_randomNumbers, 1);
	checkError(cublasGetError());
};

void Layer::makeRandom(float *target, int length){
	generateRandomNumbers();
	cublasSetVector(length, sizeof(float),randomNumbers,1,target,1);
	checkError(cublasGetError());
};
void Layer::makeRandomSmall(float *target, int length, float scale){
	float smallrand[length];
	generateRandomNumbers();
	for( int i=0 ; i<length ; i++ )
	{
		smallrand[i]=scale*randomNumbers[i];
	}
	
	cublasSetVector(length, sizeof(float),smallrand,1,target,1);
	checkError(cublasGetError());
	//delete [] zeros;
};
// er, and some other things...
void Layer::initializeWeights(){
	printf("Initializing weights...\n");
	float maxmin=0.2;
	for (i = 0; i < (inputSize/batchSize)*(outputSize/batchSize); i++) {
		//ensure they are not 0. 
		if (i<inputSize)
			input_t0[i] = (drand48()>0.5)? 0:0; // i%2;
		//weights[i] = 0.1-(0.2*(0.1 + (0.9*drand48())));
		weights[i] = maxmin-(maxmin*2*drand48());
	}
	printf("Transferring to device...\n");
	cublasSetVector(inputSize, sizeof(float), input_t0, 1, d_input_t0, 1);
	cublasSetVector((inputSize/batchSize)*(outputSize/batchSize), sizeof(weights[0]), weights, 1, d_weights, 1);
	for (i = 0; i < (inputSize/batchSize)+(outputSize/batchSize); i++) {
		weights[i] = 0.;
	}
	cublasSetVector((inputSize/batchSize), sizeof(weights[0]), weights, 1, d_visible_biases, 1);
	cublasSetVector((outputSize/batchSize), sizeof(weights[0]), weights, 1, d_hidden_biases, 1);
	printf("Generating random numbers\n");
	generateRandomNumbers();
	printf("Generated :D \n");
};

void Layer::updateWithPattern(	float* d_input_pattern){

	//cublasStatus status;
	int random_offset = 0;
	// Setup device settings
	int nBlocksForIn = inputSize/blockSize + (inputSize%blockSize == 0?0:1);
	int nBlocksForOut = outputSize/blockSize + (outputSize%blockSize == 0?0:1);
	int nBlocksForInBiases = (inputSize/batchSize)/blockSize + ((inputSize/batchSize)%blockSize == 0?0:1);
	int nBlocksForOutBiases = (outputSize/batchSize)/blockSize + ((outputSize/batchSize)%blockSize == 0?0:1);
	
	// Forward pass
	int inputs=inputSize/batchSize;
	int outputs=outputSize/batchSize;
	cublasSgemm('n','T',batchSize,outputs,inputs,
			1.f,d_input_pattern,batchSize,
			d_weights,outputs,
			0.f,d_output_p_t0,batchSize);
	probabilities<<<nBlocksForOut,blockSize>>>(d_output_p_t0, d_hidden_biases, outputSize);
	checkError(cublasGetError());
	cutoff<<<nBlocksForOut,blockSize>>>(d_output_p_t0,d_output_t0,d_randomNumbers,random_offset, outputSize);
	random_offset+=outputSize;
	checkError(cublasGetError());


	// Backwards pass
	cublasSgemm('n','n',batchSize,inputs,outputs,
//			1.f,d_output_p_t0,batchSize,
			1.f,d_output_t0,batchSize,
			d_weights,outputs,
			0.f,d_input_p_t1,batchSize);
	checkError(cublasGetError());
	// SHould do the labels here 
	/*if (numberOfLabels){
		makeRandomSmall(d_labels_totals,numberOfLabels*BATCHSIZE,0.);
		softmax<<<numberOfLabels,BATCHSIZE>>>(d_labels_out, d_labels_totals);
		arrayDivide<<<numberOfLabels,BATCHSIZE>>>(d_labels_out, d_labels_totals);	
		roulette<<<BATCHSIZE,numberOfLabels>>>(d_labels_out);	

	};
	*/
	probabilities<<<nBlocksForIn,blockSize>>>(d_input_p_t1, d_visible_biases,inputSize);//-numberOfLabels*BATCHSIZE);
	cudaThreadSynchronize();
	checkError(cublasGetError());
//	cutoff<<<nBlocksForIn,blockSize>>>(d_input_p_t1,d_input_t1,d_randomNumbers,random_offset);
	random_offset+=inputSize;

	// Forwards pass, again
	cublasSgemm('n','T',batchSize,outputs,inputs,
			1.f,d_input_p_t1,batchSize,
			d_weights,outputs,
			0.f,d_output_p_t1,batchSize);
	checkError(cublasGetError());
	probabilities<<<nBlocksForOut,blockSize>>>(d_output_p_t1, d_hidden_biases, outputSize);
	cudaThreadSynchronize();
//	cutoff<<<nBlocksForOut,blockSize>>>(d_output_p_t1,d_output_t1,d_randomNumbers,random_offset);

	cudaThreadSynchronize();
	
	biasesIncrease<<<nBlocksForInBiases,blockSize>>>(d_input_pattern, d_visible_biases, biasLearningRate/batchSize, inputSize/batchSize);
	biasesDecrease<<<nBlocksForInBiases,blockSize>>>(d_input_p_t1, d_visible_biases, biasLearningRate/batchSize, inputSize/batchSize, 0.0);

	biasesIncrease<<<nBlocksForOutBiases,blockSize>>>(d_output_p_t0, d_hidden_biases, biasLearningRate/batchSize, outputSize/batchSize);
	biasesDecrease<<<nBlocksForOutBiases,blockSize>>>(d_output_p_t1, d_hidden_biases, biasLearningRate/batchSize, outputSize/batchSize, 0.0);

	cublasSgemm('T','n',outputs,inputs,batchSize,
//			learningRate,d_output_t0,batchSize,
			learningRate,d_output_p_t0,batchSize,
			d_input_pattern,batchSize,
			1.f,d_weights,outputs);
	checkError(cublasGetError());

	cublasSgemm('T','n',outputs,inputs,batchSize,
//			-learningRate,d_output_t1,batchSize,
//			d_input_t1,batchSize,
			-learningRate,d_output_p_t1,batchSize,
			d_input_p_t1,batchSize,
			WEIGHTDECAY,d_weights,outputs);
	checkError(cublasGetError());
	generateRandomNumbers();	
};


void Layer::printGenerated(bool quiet){

	cublasGetVector(inputSize, sizeof(float), d_input_p_t1, 1, input_t0, 1);
	//cublasGetVector(inputSize, sizeof(float), d_input_t0, 1, input_t0, 1);
	if (!quiet){
		for( int i=0 ; i<inputSize	; i++ )
		{
			printf("%f |",input_t0[i]);
		}
		printf("\n");
	}

};

void Layer::setLabels(float* labels, float* position){
	if (position)
		cublasSetVector(numberOfLabels*batchSize, sizeof(float), labels, 1, position, 1);
	else
		cublasSetVector(numberOfLabels*batchSize, sizeof(float), labels, 1, d_labels_in, 1);
	checkError(cublasGetError());
};

void Layer::getLabels(float* output){	
	cublasGetVector(numberOfLabels*batchSize, sizeof(float), d_labels_out, 1, output, 1);
};

class RBM{
	/* This class will store all of the layers.
	 * 
	 * It will provide a simple interface for learning and processing.
	 *
	 * The kohonen map may be added to the end to cluster the outputs.
	 *
	 * This would probably be cleaner to add as another class.
	 *
	 */
	public:
		Layer **allLayers;
		int totalLayers;
		int *layerSizes;
		int batchSize;
		int labelSize;
		float learningRate;
		float learningRateDecay;
		float* scratch;
		float* scratch2;
	//Function declarations
	RBM(int numberOfNeuronLayers, int layersizes[], int batchsize, float learnRate, float biasLearnRate, float expDecay, int labelsize=0);
	float trainLayer(float* d_input_pattern, int layer, bool save=false);
	void trainingEpoch(float* data, int dataLength, int layer, unsigned char* labels, int epoch);
	void train(float* data, int dataLength, int totalEpochs, unsigned char* labels=NULL);
	void setLabels(float* labels, float* location=NULL);
	void getLabels(float* output);
	void calculateClassProbabilities(float* data, float *labels, bool print=false);
	void classifyAv(float* images, int* output);
	void classify(float* data, int* out, bool print=false);
	float test(float* data, int dataLength, unsigned char* labels);
};

RBM::RBM(int numberOfNeuronLayers, int layersizes[], int batchsize, float learnRate, float biasLearnRate, float expDecay, int labelsize){
	//assert(numberOfNeuronLayers>1);
	labelSize=labelsize;
	layerSizes = new int [numberOfNeuronLayers];
	for( int i=0 ; i<numberOfNeuronLayers ; i++ )
	{
		layerSizes[i]=layersizes[i];
	}
	scratch= new float [layersizes[0]*layersizes[2]];
	scratch2= new float [layersizes[0]*layersizes[2]];
	learningRate=learnRate;
	learningRateDecay=expDecay;
	batchSize=batchsize;
	totalLayers=numberOfNeuronLayers-1;
	//FUCKING NO. NOT RIGHT.
	//layerSizes[totalLayers-1]+=labelSize;
	allLayers = new Layer *[totalLayers];//(Layer*)malloc(totalLayers * sizeof(Layer));
	for( int i=0 ; i<totalLayers-1 ; i++ )
	{
		printf("New layer, layer %d - size %d by %d\n",i,layerSizes[i],layerSizes[i+1]);
		allLayers[i]=new Layer(layerSizes[i],layerSizes[i+1], batchSize, learnRate, biasLearnRate);
	}
	//Special case for the last one.
	printf("New layer, layer %d - size %d by %d\n",totalLayers-1,layerSizes[totalLayers-1],layerSizes[totalLayers]);
	allLayers[totalLayers-1]=new Layer(layerSizes[totalLayers-1],layerSizes[totalLayers], batchSize, learnRate, biasLearnRate, labelSize);

};

void RBM::setLabels(float* labels, float* location){
	//if (location)

	//else
	allLayers[totalLayers-1]->setLabels(labels,allLayers[totalLayers-1]->d_labels_in);
};

void RBM::getLabels(float* output){
	allLayers[totalLayers-1]->getLabels(output);
};


float RBM::trainLayer(float* d_input_pattern, int layer, bool save){
	
	//Small speedup, there is a special case when layer==0
	if (layer==0){
		if (save) cublasGetVector(layerSizes[0]*BATCHSIZE, sizeof(float), allLayers[0]->d_input_t0,1,scratch,1);
		allLayers[0]->updateWithPattern(d_input_pattern);
		if (save) cublasGetVector(layerSizes[0]*BATCHSIZE, sizeof(float), allLayers[0]->d_input_p_t1,1,scratch2,1);
		if (save) return difference(scratch,scratch2,layerSizes[0]*BATCHSIZE)/(float)(BATCHSIZE*layerSizes[0]);
		return 0.;
	}
	//Otherwise, put the input pattern into the bottom layer
	allLayers[0]->pushUp(d_input_pattern,allLayers[1]->d_input_t0,false);

	//Push up through the rest
	for( int i=1 ; i<layer ; i++ )
	{
		allLayers[i]->pushUp(allLayers[i]->d_input_t0,allLayers[i+1]->d_input_t0,false);	
	}
	
	//Now train that layer
	if (save) cublasGetVector(layerSizes[layer]*32, sizeof(float), allLayers[layer]->d_input_t0,1,scratch,1);
	allLayers[layer]->updateWithPattern(allLayers[layer]->d_input_t0);
	if (save) cublasGetVector(layerSizes[layer]*32, sizeof(float), allLayers[layer]->d_input_p_t1,1,scratch2,1);
	if (save) return difference(scratch,scratch2,layerSizes[layer]*BATCHSIZE)/(float)(BATCHSIZE*layerSizes[layer]);
	return 0.;	
};

void RBM::trainingEpoch(float* data, int dataLength, int layer, unsigned char* labels, int epoch){
		int inputSize=layerSizes[0];
		int labelSize=allLayers[totalLayers-1]->numberOfLabels;
		float imageBatch[inputSize*batchSize];
		float labelsArray[labelSize*batchSize];
		float labelsArrayColMajor[labelSize*batchSize];
		//printf("Total size of input:%d\n",inputSize*batchSize);
		//printf("Total size of labels:%d\n",labelSize*batchSize);
		float diff=0.;
		bool save=false;
		FILE *f;
		if (save){
			char name[50];
			sprintf(name,"/home/ian/project/mnistreconserrorL%d-E%d.dat",layer,epoch);
			f=fopen(name,"a");
			fprintf(f,"Currentrun	ReconstructionError\n");
		}
		for (int runOverFile=batchSize; runOverFile<dataLength; runOverFile+=batchSize){
			convertToColumnMajor(&data[(runOverFile-batchSize)*inputSize],imageBatch,batchSize*inputSize,batchSize);			
			cublasSetVector(inputSize*batchSize, sizeof(float), imageBatch, 1, allLayers[0]->d_input_t0, 1);
			if (layer==totalLayers-1){
				//memset(labelsArray,0,labelSize*batchSize);	
				
				for( int i=0 ; i<batchSize ; i++ )
				{
					for( int j=0 ; j<labelSize ; j++ )
					{
						if ((j==(int)labels[(runOverFile-batchSize)+i])){
							labelsArray[i*labelSize + j]=1.0;
							//printf("Setting <%d> - Image %d=%d\n",i*labelSize+j,(runOverFile-batchSize)+i,(int)labels[(runOverFile-batchSize)+i]);
						}
						else{
							labelsArray[i*labelSize + j]=0.0;
						}
					}
				}
				convertToColumnMajor(labelsArray,labelsArrayColMajor,labelSize*batchSize,batchSize);
				setLabels(labelsArrayColMajor);
			}
			cudaThreadSynchronize();
			diff=trainLayer(allLayers[0]->d_input_t0,layer,save);
			if (save) fprintf(f,"%d	%f\n",runOverFile,diff);
		}
		if (save) fclose(f);
};

void RBM::train(float* data, int dataLength, int totalEpochs, unsigned char* labels){

	for( int layer=0 ; layer<totalLayers ; layer++ )
	{
		printf("Layer: %d/%d\n",layer,totalLayers);
		for( int epoch=0 ; epoch<totalEpochs ; epoch++ )
		{
			printf("   Epoch: %d/%d\n",epoch,totalEpochs);
			trainingEpoch(data,dataLength,layer,labels,epoch);
			allLayers[layer]->learningRate*=learningRateDecay;	
		}
		
	}
};

void RBM::classifyAv(float* images, int* output){
	int inputSize=layerSizes[0];
	float imagesColMajor[inputSize*batchSize];
	int topLabel;
	float topLabelProbability;
	int labelSize=allLayers[totalLayers-1]->numberOfLabels;
	float labels[labelSize*batchSize];
	float labelsAverage[labelSize*batchSize];
	float zeros[labelSize*batchSize];
	float total=0.;

	for( int i=0 ; i<labelSize*batchSize ; i++ )
	{
		zeros[i]=0.f;
		labelsAverage[i]=0.f;
	}
	convertToColumnMajor(images,imagesColMajor,batchSize*inputSize,batchSize);			
	cublasSetVector(inputSize*batchSize, sizeof(float), imagesColMajor, 1, allLayers[0]->d_input_p_t1, 1);
	setLabels(zeros,allLayers[totalLayers-1]->d_labels_out);
	for( int layer=0 ; layer<(totalLayers-1) ; layer++ )
	{
		allLayers[layer]->generateRandomNumbers();
		allLayers[layer]->pushUp(allLayers[layer]->d_input_p_t1, allLayers[layer+1]->d_input_p_t1, false);
	}
	allLayers[totalLayers-1]->pushUp(allLayers[totalLayers-1]->d_input_p_t1,allLayers[totalLayers-1]->d_output_t0,false);
	allLayers[totalLayers-1]->pushDown(allLayers[totalLayers-1]->d_output_t0,false);
	getLabels(labels);

	setLabels(zeros,allLayers[totalLayers-1]->d_labels_out);

	for( int i=0 ; i<10 ; i++ )
	{
		allLayers[totalLayers-1]->generateRandomNumbers();
		allLayers[totalLayers-2]->pushUp(allLayers[totalLayers-2]->d_input_p_t1,allLayers[totalLayers-1]->d_input_p_t1,false);
		allLayers[totalLayers-1]->pushUp(allLayers[totalLayers-1]->d_input_p_t1,allLayers[totalLayers-1]->d_output_t0,true);
		allLayers[totalLayers-1]->pushDown(allLayers[totalLayers-1]->d_output_t0,false);
		getLabels(labels);

		setLabels(zeros,allLayers[totalLayers-1]->d_labels_out);
		for( int batch=0 ; batch<batchSize ; batch++ )
		{
			//Softmax, find total
			total=0.0;
			for( int labelNumber=0; labelNumber<labelSize ; labelNumber++ )
			{
				total+=labels[ (labelNumber*batchSize) + batch ];
			}
			//Normalise
			if( total>0. )
			{
				for( int labelNumber=0; labelNumber<labelSize ; labelNumber++ )
				{
		//			labels[ (labelNumber*batchSize) + batch ]/=total;
				}
			}
			else {
				printf("Not above zero, eh?\n");
			};
			//Now average
			for( int labelNumber=0; labelNumber<labelSize ; labelNumber++ )
			{
				labelsAverage[ (labelNumber*batchSize) + batch ]+=labels[ (labelNumber*batchSize) + batch ];
			}
		}
	}
	
	//Ok, now they've been averaged over 10 iterations, find the largest and return
	for( int batch=0 ; batch<batchSize ; batch++ )
	{
		topLabelProbability=-1.;
		for( int labelNumber=0; labelNumber<labelSize ; labelNumber++ )
		{
			if (labelsAverage[ (labelNumber*batchSize) + batch ]>topLabelProbability){
				topLabelProbability=labelsAverage[labelNumber];
				topLabel=labelNumber;
			}
		}
		output[batch]=topLabel;
		
	}
	

};

void RBM::calculateClassProbabilities(float* data, float *labels, bool print){

	int inputSize=layerSizes[0];
	int labelSize=allLayers[totalLayers-1]->numberOfLabels;
	int batchSize=BATCHSIZE;
	int toplayer=totalLayers-1;
	float imageBatch[batchSize*inputSize];
	for( int i=0 ; i<batchSize*labelSize ; i++ )
	{
		labels[i]=0.f;
	}
	
	convertToColumnMajor(data,imageBatch,batchSize*inputSize,batchSize);			
	
	cublasSetVector(batchSize*inputSize, sizeof(float), imageBatch, 1, allLayers[0]->d_input_p_t1, 1);

	allLayers[toplayer]->generateRandomNumbers();

	allLayers[toplayer]->setLabels(labels,allLayers[toplayer]->d_labels_out);
	allLayers[toplayer]->setLabels(labels,allLayers[toplayer]->d_labels_in);
	
	for( int layer=0 ; layer<toplayer ; layer++ )
	{	
		allLayers[layer]->pushUp(allLayers[layer]->d_input_p_t1,allLayers[layer+1]->d_input_p_t1,false);
	}
	
	allLayers[toplayer]->pushUp(allLayers[toplayer]->d_input_p_t1,allLayers[toplayer]->d_output_t0,false);
	allLayers[toplayer]->pushDown(allLayers[toplayer]->d_output_t0,false);
	getLabels(labels);

};

void RBM::classify(float* data, int *out, bool print){

print=false;	
	int labelSize=allLayers[totalLayers-1]->numberOfLabels;
	int batchSize=BATCHSIZE;
	float labels[batchSize*labelSize];
	for( int i=0 ; i<batchSize*labelSize ; i++ )
	{
		labels[i]=0.f;
	}
	
	calculateClassProbabilities(data,labels);

	float topVal=-1.;
	int actual;
	for( int batch=0 ; batch<batchSize ; batch++ )
	{
		if( print )
			printf("\n\nBatchNumber: %d\n", batch);
		topVal=labels[batch];
		actual=0;
		for( int i=0 ; i<labelSize ; i++ )
		{
			if(labels[batchSize*(i)+batch]>topVal){
				topVal=labels[batchSize*(i)+batch];
				actual=i;
			}
			if (print)
				printf("%d=%f  ",i,labels[batchSize*(i)+batch]);
		}
		out[batch]=actual;
	}
	

};


float RBM::test(float* data, int dataLength, unsigned char* labels){
	int labelsOut[batchSize];
	int inputSize=layerSizes[0];
	int correct=0;
	int runOverFile;
	for (runOverFile=batchSize; runOverFile<dataLength; runOverFile+=batchSize){
		classify(&data[(runOverFile-batchSize)*inputSize],labelsOut,(runOverFile<60));			
		for( int i=0 ; i<batchSize ; i++ )
		{
			if( ((int)labels[(runOverFile-batchSize)+i])==labelsOut[i] )
			{
				correct++;
			}
			else{
		//		printf("%d - actual %d reported %d\n",runOverFile-batchSize+i,((int)labels[(runOverFile-batchSize)+i]),labelsOut[i]);
			};
		}
	};
	return (float)correct/(float)runOverFile;
};


void classifyMultiple(RBM** rbms, int numRBMs, float* data, int* labels){
	
	int batchSize=rbms[0]->batchSize;
	int labelSize=rbms[0]->labelSize;
	
	float labelAverages[batchSize*labelSize];
	float labelSingle[batchSize*labelSize];
	
	bool norm=false;//true;
	bool logprob=false;
	float total;

	for( int i=0 ; i<labelSize*batchSize ; i++ )
	{
		labelSingle[i]=0.f;
		labelAverages[i]=0.f;
	}
	for( int rbm=0 ; rbm<numRBMs ; rbm++ )
	{
		rbms[rbm]->calculateClassProbabilities(data,labelSingle);
		//Normalise?
		if (norm)
			for( int batch=0 ; batch<batchSize ; batch++ )
			{
				total=0.0001;
				for( int i=0 ; i<labelSize ; i++ )
					total+=labelSingle[batchSize*(i)+batch];
				for( int i=0 ; i<labelSize ; i++ )
					labelSingle[batchSize*(i)+batch]/=total;
			};

		for( int i=0 ; i<labelSize*batchSize ; i++ )
		{
			if (logprob)
				labelAverages[i]+=log(labelSingle[i]);
			else
				labelAverages[i]+=labelSingle[i];
		}
	}
	float topVal=-1.;
	int actual;
	for( int batch=0 ; batch<batchSize ; batch++ )
	{
		topVal=labelAverages[batch]-1.;
		for( int i=0 ; i<labelSize ; i++ )
		{
			if(labelAverages[batchSize*(i)+batch]>topVal){
				topVal=labelAverages[batchSize*(i)+batch];
				actual=i;
			}
		}
		labels[batch]=actual;
	}
};


float testMultiple(RBM** rbms, int numRBMs, float* data, int dataLength, unsigned char* labels){
	int batchSize=rbms[0]->batchSize;
	int labelsOut[batchSize];
	int inputSize=rbms[0]->layerSizes[0];
	int correct=0;
	int runOverFile;
	for (runOverFile=batchSize; runOverFile<dataLength; runOverFile+=batchSize){
		classifyMultiple(rbms, numRBMs, &data[(runOverFile-batchSize)*inputSize],labelsOut);			
		for( int i=0 ; i<batchSize ; i++ )
		{
			if( ((int)labels[(runOverFile-batchSize)+i])==labelsOut[i] )
			{
				correct++;
			}
			else{
				//printf("actual %d reported %d\n",((int)labels[(runOverFile-batchSize)+i]),labelsOut[i]);
			};
		}
	};
	return (float)correct/(float)runOverFile;
};


struct header {
	int magic; // 4 bytes
	int ndim; // 4 bytes, little endian
	int dim[3];
};
int checkKeyboardNumber(){
	if( key[KEY_0] )
		return 0;
	if( key[KEY_1] )
		return 1;
	if( key[KEY_2] )
		return 2;
	if( key[KEY_3] )
		return 3;
	if( key[KEY_4] )
		return 4;
	if( key[KEY_5] )
		return 5;
	if( key[KEY_6] )
		return 6;
	if( key[KEY_7] )
		return 7;
	if( key[KEY_8] )
		return 8;
	if( key[KEY_9] )
		return 9;
	return -1;
	
};
/* Main */
int main(int argc, char** argv)
{    

	cublasStatus status;
    
    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

/// SETTINGS    ----------------------------------------------

	int inputsqrt=28;//32;//atoi(argv[1]);
	int numTraining=58000;//46800;
	int numTesting=10000;//46800;
	float norb=false;
	int totalAmount=60000;
	
if (norb){
	inputsqrt=32;
	numTraining=46800;
	numTesting=46800;
	totalAmount=46800;
	
}

int trainingFilesize=totalAmount*inputsqrt*inputsqrt*sizeof(float);
int testingFilesize=totalAmount*inputsqrt*inputsqrt*sizeof(float);

	int outputVectorSize=32;//atoi(argv[2]);
	int batchsize=BATCHSIZE;//atoi(argv[4]);
	int totalLayers=4;
	int layerAcc=0;
	int sizes[20]={inputsqrt*inputsqrt,512,512,2048,outputVectorSize};
	//int sizes[5]={inputsqrt*inputsqrt,256,256,1024,outputVectorSize};
float decayValue=1.;//atof(argv[4]);	
RBM **beliefNets;
char* outfilename=NULL;
int totalNets=1;
float learningRate;
float biasLearningRate=0.f;
int totalEpochs=1;
int trainingAmount;
//read in from cli
	int c;
	opterr = 0;

	char delims[] = " ";
	char *result = NULL;
	while ((c = getopt (argc, argv, "b:r:o:a:e:l:d:n:")) != -1)
		switch (c)
		{
			case 'o':
				outfilename = optarg;
			case 'a':
				trainingAmount = atoi(optarg);
				break;
			case 'e':
				totalEpochs = atoi(optarg);
				break;
			case 'l':
				learningRate = atof(optarg);
				break;
			case 'b':
				biasLearningRate = atof(optarg);
				break;
			case 'd':
				decayValue = atof(optarg);
				break;
			case 'r':
				totalNets = atoi(optarg);
				break;	
			case 'n':
				result = strtok( optarg, delims );
				while( result != NULL ) {
					printf( "layersizes %d - %d \n", layerAcc++, atoi(result) );
					sizes[layerAcc]=atoi(result);
					result = strtok( NULL, delims );
				}
				totalLayers = layerAcc+1;
				break;
			case '?':
				return 1;
			default:
				abort ();
		}




#ifdef DISPLAY
	if (allegro_init() != 0)
		return 1;
	
	install_mouse();
	install_keyboard();
//	show_os_cursor(1);
enable_hardware_cursor();
show_os_cursor(0);
	set_color_depth(24);
	if (set_gfx_mode(GFX_AUTODETECT_WINDOWED, WIDTH, HEIGHT, 0, 0) != 0) {
		if (set_gfx_mode(GFX_SAFE, WIDTH, HEIGHT, 0, 0) != 0) {
			set_gfx_mode(GFX_TEXT, 0, 0, 0, 0);
			allegro_message("Unable to set any graphic mode\n%s\n", allegro_error);
			return 1;
		}
	}
	BITMAP *buffer;

	buffer = create_bitmap(SCREEN_W, SCREEN_H);
	set_palette(desktop_palette);
#endif
srand48(time(NULL));	
	//int count=0;
	for( int i=0 ; i<10 ; i++ )
	{
		printf("%f \n",drand48());
	}

beliefNets = new RBM *[totalNets];
for( int net=0 ; net<totalNets ; net++ )
{	
	printf("Creating RBM %d/%d\n",net+1,totalNets);	
	beliefNets[net] = new RBM(totalLayers,sizes,batchsize, learningRate,biasLearningRate, decayValue,10);
	printf("%f - learnding rate\n",learningRate);	
	printf("/Creating RBM\n");	
}

//RBM rbm = RBM(4,sizes,batchsize, atof(argv[1]),decayValue,10);
//int epochSize=1000;	
//int l=0;
// read and show an image
    FILE *f;
//unsigned char pixelValue;	

 /////                             LABELS           ////////////////	
unsigned char labels[numTraining];
unsigned char testLabels[numTesting];
if (norb)
	f=fopen("/home/ian/project/data/smallNorb-Training-labels.uchar","r");
else
	f=fopen("/home/ian/project/data/mnist-labels-58k.uchar","r");
fread(&labels,sizeof(unsigned char),numTraining,f);
fclose(f);
printf("Read training labels\n");

if (norb)
	f=fopen("/home/ian/project/data/smallNorb-Testing-labels.uchar","r");
else
	f=fopen("/home/ian/project/data/mnist-labels-t10k.uchar","r");
fread(&testLabels,sizeof(unsigned char),numTesting,f);
fclose(f);
printf("Read testing labels\n");

//Print out a few labels
 
for( int i=0 ; i<10 ; i++ )
{
	printf("%d %d\n",(int)labels[i], (int)testLabels[i]);
}


/////                             IMAGES            ////////////////	



//Read in the training file
int ftrain;

float *data;
if (norb)
	ftrain = open("data/smallNorb-Training-images32x32.float",O_RDONLY);
else
	ftrain = open("data/mnist-train.float", O_RDONLY);
if (!f)
	printf("File not found\n");

data = (float *) mmap(0, trainingFilesize, PROT_READ, MAP_SHARED, ftrain, 0);



// REad in the test file

float * testData;
int ftest;
if (norb)
	ftest = open("data/smallNorb-Testing-images32x32.float",O_RDONLY);
else
	ftest = open("data/mnist-test.float", O_RDONLY);
if (!f)
	printf("File not found\n");

testData = (float *) mmap(0, testingFilesize, PROT_READ, MAP_SHARED, ftest, 0);

// **************************************
// The actual training


int testStart, testSize;

if (norb){
	testStart=46800-2340;
	testSize=2340;
}
else{
	testStart=55000;
	testSize=3000;
}

time_t start, end;




time(&start);
for( int net=0 ; net<totalNets ; net++ )
{
	printf("------%d / %d--------\n",net+1,totalNets);
	beliefNets[net]->train(data,trainingAmount,totalEpochs,labels);
}
time(&end);
float timetaken=difftime(end,start);
float timetakenTraining=difftime(end,start);
printf("Time taken for training: %fs\n Images per second: %f\n",timetaken,(totalNets*totalEpochs*trainingAmount)/timetaken); 

FILE* fout;
fout=fopen(outfilename,"a+");
printf("Testing...\n");
fprintf(fout,"Sizes: ");
for( int i=1 ; i<totalLayers ; i++ )
{
	fprintf(fout, "%d ",sizes[i]);
}

//fprintf(fout,"%d %f %f %f ",atoi(argv[3]),atof(argv[1]), decayValue, testMultiple(beliefNets,totalNets,testData,10000,testLabels));
fprintf(fout,"iterations: %d fractionalEpochs %f LR: %f amount: %d Decay: %f biasLR: %f TrainingError: %f TestError: %f Validationerror: %f | ",totalEpochs,((float)totalEpochs*trainingAmount)/58000.,learningRate,trainingAmount, decayValue,biasLearningRate, 
		testMultiple(beliefNets,totalNets,data,trainingAmount,labels),
		testMultiple(beliefNets,totalNets,&data[inputsqrt*inputsqrt*testStart],testSize,&labels[testStart]),
		testMultiple(beliefNets,totalNets,testData,numTesting,testLabels)	
		);

time(&start);
for( int i=0 ; i<10 ; i++ )
for (int l=0;l<totalNets;l++)
	fprintf(fout," %f", beliefNets[l]->test(testData,numTesting,testLabels));
time(&end);
timetaken=difftime(end,start);

fprintf(fout," timeTraining: %f size: %d weightdecay: %f ", timetakenTraining, (int)BATCHSIZE, (float)WEIGHTDECAY);
fprintf(fout,"testingrate: %f",10*(totalNets*numTesting)/timetaken); 
fprintf(fout,"\n");

printf("Done :)\n");
fclose(fout);
if (munmap(data, trainingFilesize) == -1) {
		perror("Error un-mmapping the file");
		    }
close(ftrain);
if (munmap(testData, testingFilesize) == -1) {
		perror("Error un-mmapping the file");
		    }
close(ftest);

// **************************************
#ifdef DISPLAY

float labelsIn[10*32];
float labelsOutColMajor[10*32];
for( int i=0 ; i<10 ; i++ )
{
	for( int j=0 ; j<32 ; j++ )
	{
		if( i==j )
		{
			labelsIn[10*j +i]=0.0;
		}
		else{
			labelsIn[10*j + i]=0.0;
	
		}
	}
}
float randscale=1.0;
convertToColumnMajor(labelsIn,labelsOutColMajor,10*32,32);
float scratch[sizes[0]*sizes[1]];
int theNumber=0;
install_timer();
int batchNum=0;
int generateNumber=0;
RBM* rbm = beliefNets[0];
	//while (currentRun<100){
	while (!key[KEY_ESC]){
		//printf("Current run: %d\n",currentRun);
		if (key[KEY_UP]){
			randscale+=0.01;
				printf("randscale %f\n",randscale);
			//theNumber++;
			if( theNumber>31 )
			{
				theNumber=0;
				batchNum++;
			}
			rest(100);
		}
		if (key[KEY_DOWN]){
			randscale-=0.01;
				printf("randscale %f\n",randscale);
			rest(100);
		}
		generateNumber=checkKeyboardNumber();
		if( generateNumber>-1 )
		{
			printf("New number = %d\n",generateNumber);
			for( int i=0 ; i<10 ; i++ )
			{
				labelsIn[i]=0.;
			}
			labelsIn[generateNumber]=1.;
			convertToColumnMajor(labelsIn,labelsOutColMajor,10*32,32);
			//setrandom too
		}
		rbm->allLayers[2]->makeRandomSmall(rbm->allLayers[2]->d_input_p_t1,512*32,0*randscale);	
		
		rbm->allLayers[2]->setLabels(labelsOutColMajor,rbm->allLayers[2]->d_labels_out);
		rbm->allLayers[2]->setLabels(labelsOutColMajor,rbm->allLayers[2]->d_labels_in);
		
		rbm->allLayers[2]->pushUp(rbm->allLayers[2]->d_input_p_t1,rbm->allLayers[2]->d_output_p_t0,false);
		rbm->allLayers[2]->generateRandomNumbers();
		cublasGetVector(2048*32, sizeof(float), rbm->allLayers[2]->d_output_p_t0,1,scratch,1);
		drawImage(buffer,10,10,4,64,32,scratch,32,theNumber,1,15);	
		cublasGetVector(10*32, sizeof(float), rbm->allLayers[2]->d_labels_in,1,scratch,1);
		drawImage(buffer,200,10,8,10,1,scratch,32,theNumber,1,15);	
		
		rbm->allLayers[2]->pushDown(rbm->allLayers[2]->d_output_p_t0,false);
		cublasGetVector(512*32, sizeof(float), rbm->allLayers[2]->d_input_p_t1,1,scratch,1);
		drawImage(buffer,10,300,2,32,16,scratch,32,theNumber,0,0);	
		
		rbm->allLayers[1]->pushDown(rbm->allLayers[2]->d_input_p_t1,false);
		cublasGetVector(512*32, sizeof(float), rbm->allLayers[1]->d_input_p_t1,1,scratch,1);
		drawImage(buffer,10,400,4,32,16,scratch,32,theNumber,1,15);	
		
		rbm->allLayers[0]->pushDown(rbm->allLayers[1]->d_input_p_t1,false);
		cublasGetVector(28*28*32, sizeof(float), rbm->allLayers[0]->d_input_p_t1,1,scratch,1);
		drawImage(buffer,10,600,4,28,26,scratch,32,theNumber,0,0);	
		blit(buffer, screen, 0, 0, 0, 0, SCREEN_W, SCREEN_H);
		//find position
	}

#endif
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
