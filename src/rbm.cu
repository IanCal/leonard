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

#include "rbm.cuh"


/* 
 * The actual code for the RBM goes in here
 */

RBM::pushDown(int layer, bool input_t0, bool output_t0, bool useProbabilities){

	//Basic variables
	
	int inputs = layerSizes[layer];
	int outputs = layerSizes[layer+1];
	int inputBatchSize = inputs * batchSize;
	int outputBatchSize = outputs * batchSize;
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

RBM::pushUp(int layer, bool input_t0, bool output_t0, bool useProbabilities){

	//Basic variables
	
	int inputs = layerSizes[layer]+labelSizes[layer];
	int outputs = layerSizes[layer+1];
	int inputBatchSize = inputs * batchSize;
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
	
	//cutoff kernel
	cutoff<<<numberOfBlocks,blockSize>>>(d_output_p, d_outout, d_randomNumbers, outputBatchSize);

};

RBM::alternatingGibbsSampling(int layer, int iterations, bool probabilisticInput, bool probabilisticOutput, bool startAtTop){

	// Push up the initial pattern, then down to the inputs
	if (!startAtTop)
		pushUp(layer, true, true, probabilisticInput);
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

RBM::updateWeightsInLayer(int layer){

	// Update the visible biases
	biasesIncrease<<<nBlocksForInBiases,blockSize>>>(d_input_pattern, d_visible_biases, biasLearningRate, inputSize/batchSize);
	biasesDecrease<<<nBlocksForInBiases,blockSize>>>(d_input_p_t1, d_visible_biases, biasLearningRate, inputSize/batchSize, 0.0);

	// Update the hidden biases
	biasesIncrease<<<nBlocksForOutBiases,blockSize>>>(d_output_p_t0, d_hidden_biases, biasLearningRate, outputSize/batchSize);
	biasesDecrease<<<nBlocksForOutBiases,blockSize>>>(d_output_p_t1, d_hidden_biases, biasLearningRate, outputSize/batchSize, 0.0);

	// Update the weights
	cublasSgemm('T','n',outputs,inputs,batchSize,
			learningRate,d_output_p_t0,batchSize,
			d_input_pattern,batchSize,
			1.f,d_weights,outputs);
	checkError(cublasGetError());

	cublasSgemm('T','n',outputs,inputs,batchSize,
			-learningRate,d_output_p_t1,batchSize,
			d_input_p_t1,batchSize,
			weightDecay,d_weights,outputs);
	checkError(cublasGetError());

};

RBM::updateWeights(){

	int topRequiredLayer=0;

	// We need to know the top layer with a learning rate
	for( int layer=0 ; layer<layers ; layer++ )
	{
		if( learningRate[layer]!=0.0 )
		{
			topRequiredLayer=layer;
		}
	}
	

	for( int layer=0 ; layer<topRequiredLayer; layer++ ){
		if( learningRate[layer]==0 ){
			pushUp(layer, true, true);
		}
		else{
			alternatingGibbsSampling(layer, CDSamples);
			updateWeightsInLayer(layer);
		}
	}

};

RBM::setInputPattern(){
	inputSource->nextInput(d_input_pt0[0], layerSizes[0]+labelSizes[0]);
};

RBM::learningIteration(){
	
	setInputPattern();
	updateWeights();
	// Could be put outside of this, make it easier to pass
	// current run number, layer, etc.
	// Or maybe the controller should keep track of this...
	// Yes, because it's the only one that knows what's happening.
	parameterUpdater->updateParameters(this);
};
