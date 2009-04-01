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
/* Matrix size */
#define N  (1000)
#define blockSize (512)
#define WIDTH   800
#define HEIGHT  800
#define BATCHSIZE 32 
#define WEIGHTDECAY 1.0

#include "randomNumbers.cu"


/**
 * Convert overall energies into probabilities based on the sigmoid function.
 * @param neurons The neurons to operate on.
 * @param biases The biases of the neurons. These must be the same size.
 * @param maxLength This is the number of neurons.
 */
__global__ void probabilities( float* neurons, float* biases, int maxLength){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxLength){
		neurons[idx] = 1./(1.+exp(-biases[idx/BATCHSIZE]-neurons[idx]));
	}
};

/**
 * Set all elements of an array to a single value. 
 * @param input The array to work on.
 * @param maxLength The total length of the array.
 * @param value A floating point value to set the array to.
 */
__global__ void setToValue( float* input, int maxLength, float value){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxLength){
		input[idx]=value;
	}
};
__global__ void setRandomScale( float* input, float* random, int maxLength, float scale){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxLength){
		input[idx]=(scale/2.)-random[idx]*scale;
	}
};

__global__ void biasesIncrease(float* in, float* out, float learningRate, int maxLength, int batchsize){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxLength)
		for( int i=0 ; i<batchsize ; i++ )
		{
			out[idx]+=in[idx*batchsize + i]*learningRate;
		}	
};

__global__ void biasesDecrease(float* in, float* out, float learningRate, int maxLength, int batchsize, float sparsity){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxLength)
		for( int i=0 ; i<batchsize ; i++ )
		{
			out[idx]-=in[idx*batchsize + i]*learningRate + sparsity;
		}	
};

__global__ void cutoff( float* neurons_in, float* neurons_out, unsigned int* seed_m0, unsigned int* seed_m1, int maxLength){
    
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float random; 
	seed_m0[idx] = 18000u * (seed_m0[idx] & 0xFFFFu) + (seed_m0[idx] >> 16);
	seed_m1[idx] = 30903u * (seed_m1[idx] & 0xFFFFu) + (seed_m1[idx] >> 16);
	random = static_cast<float>((seed_m0[idx] << 16) + (seed_m1[idx] & 0xFFFFu)) / 4294967296.0f;
	if (idx<maxLength)
		neurons_out[idx]= (random < neurons_in[idx]) ? 1. : 0.;
};


// Currently unused:
__global__ void setRand( float* input, int maxLength, Rand48 rng){
    
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxLength){
		rand48_loadState(rng);
		input[idx]= rand48_nextFloat(rng);
		rand48_storeState(rng);
	}
};

__global__ void softmax(float* in, float* out){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	in[idx]=expf(in[idx]);
	out[idx%BATCHSIZE]+=in[idx];
};
__global__ void arrayDivide(float* in, float* out){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (out[idx%BATCHSIZE]>0.)
		in[idx]/=out[idx%BATCHSIZE];
};
/*
__global__ void roulette(float* in){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
//	if (out[idx/numLabels]>0.)
//		in[idx]/=out[idx/numLabels];
};
*/


