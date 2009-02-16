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


RBM::pushUp(int layer, bool input_t0, bool output_t0){

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
	if (input_t0)
		d_input = d_input_t0[layer];	
	else
		d_input = d_input_ptn[layer];

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

RBM::alternatingGibbsSampling(int layer, int iterations, bool stochasticInput, bool stochasticOutput, bool startAtTop){

	// Push up the initial pattern, then down to the inputs
	if (!startAtTop)
		pushUp(layer, true, true);
	pushDown(layer, false, true);
	//Cycle doing this
	for( int i=0 ; i<iterations-1 ; i++ )
	{
		pushUp(layer, false, false);
		pushDown(layer, false, false);
	}
	//Final push up.	
	pushUp(layer, false, false);
	
};

RBM::updateWeightsInLayer(int layer){

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
