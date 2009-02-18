
/* Matrix size */
#define N  (1000)
#define blockSize (512)
#define WIDTH   800
#define HEIGHT  800
#define BATCHSIZE 512
#define WEIGHTDECAY 1.0

//need to add temperature to this really for sim allealing.
__global__ void probabilities( float* neurons, float* biases, int maxLength){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxLength){
		neurons[idx] = 1./(1.+exp(-biases[idx/BATCHSIZE]-neurons[idx]));
		//neurons[idx] = 0.5+tanh(biases[idx/BATCHSIZE]+neurons[idx]);
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
__global__ void biasesIncrease(float* in, float* out, float learningRate, int maxLength){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//out[idx]=0.f;
	if (idx<maxLength)
	for( int i=0 ; i<BATCHSIZE ; i++ )
	{
		out[idx]+=in[idx*BATCHSIZE + i]*learningRate;
	}	
};

__global__ void biasesDecrease(float* in, float* out, float learningRate, int maxLength, float sparsity){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//out[idx]=0.f;
	if (idx<maxLength)
	for( int i=0 ; i<BATCHSIZE ; i++ )
	{
		out[idx]-=in[idx*BATCHSIZE + i]*learningRate + sparsity;
	}	
};

__global__ void cutoff( float* neurons_in, float* neurons_out, float* random, int maxLength){
    
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxLength)
	neurons_out[idx]= (random[idx] < neurons_in[idx]) ? 1. : 0.;
	//neurons_out[idx]=neurons_in[idx];
};
