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
#include "BasicFileInput.h"

//mmapping
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

BasicFileInput::BasicFileInput(char *imagesFileName, char *labelsFileName, int length)
{
	//Set the basic variables
	currentItem = 0;
	totalItems = length;
	initialised = false;
	this->imagesFileName=imagesFileName;
	this->labelsFileName=labelsFileName;

}

void BasicFileInput::setImagesFile(char *filename, int length, int itemSize, int batchSize)
{

	//Need to initialise the array that will store the image
	inputColumnMajor = (float*) malloc(batchSize*itemSize * sizeof(float));	
	if (!inputColumnMajor){
		printf("Could not allocate array of size %dMB for input file, dying\n",(length * sizeof(float))/1e6);	
		exit(EXIT_FAILURE);
	}

	inputFile = open(imagesFileName,O_RDONLY);
	if (inputFile == -1){
		printf("Error opening the file %s", imagesFileName);
		exit(EXIT_FAILURE);
	}

	//Now mmap it so we don't need the whole file in memory at one time

	imagesFileMap = (float *) mmap(0, length*itemSize*sizeof(float), PROT_READ, MAP_SHARED, inputFile, 0);
	if (imagesFileMap == MAP_FAILED) {
		close(inputFile);
		printf("Error mmapping the file %s", imagesFileName);
		exit(EXIT_FAILURE);
	}
};

void BasicFileInput::setLabelsFile(char *filename, int fileLength, int labelSize, int layers, int batchSize)
{
	labelsColumnMajor = new float*[layers];
	for( int i=0 ; i<layers ; i++ )
	{
		labelsColumnMajor[i]= new float[batchSize*labelSize];
		for( int j=0 ; j<labelSize*batchSize ; j++ )
		{
			labelsColumnMajor[i][j]=0.0f;
		}
		
	}
	allLabels = new unsigned char[fileLength];
	FILE *f=fopen(filename,"r");
	fread(allLabels,sizeof(unsigned char),fileLength,f);
	fclose(f);
};


float* BasicFileInput::getNextInput(RBM *currentRBM)
{
	if (!initialised){
		initialise(currentRBM);
		initialised = true;
	}
	int batchSize = currentRBM->batchSize;
	int inputSize = currentRBM->layerSizes[0] - currentRBM->labelSizes[0];
	int currentPosition = currentItem*inputSize;
	for( int batch=0 ; batch<batchSize ; batch++ )
	{
		for( int i=0 ; i<inputSize ; i++ )
		{
			inputColumnMajor[batch+(i*batchSize)]=imagesFileMap[(currentPosition+i)+inputSize*batch];
		}
	}
	iteratedOverImages=true;
	if (iteratedOverLabels){
		currentItem+=batchSize;
		if (currentItem>=totalItems-batchSize)
			currentItem=0;
		iteratedOverLabels=false;
		iteratedOverImages=false;
	}
	return inputColumnMajor;
};

float** BasicFileInput::getNextLabel(RBM *currentRBM)
{
	if (!initialised){
		initialise(currentRBM);
		initialised = true;
	}
	int batchSize = currentRBM->batchSize;
	int labelSize = currentRBM->labelSizes[currentRBM->numberOfNeuronLayers-2];
	int topLayer = currentRBM->numberOfNeuronLayers-2;

	for( int batch=0 ; batch<batchSize ; batch++ )
	{
		for( int i=0 ; i<labelSize ; i++ )
		{
			if( (int)allLabels[currentItem+batch]==i )
			{
				labelsColumnMajor[topLayer][batch+(i*batchSize)]=1.0;
			}
			else
			{
				labelsColumnMajor[topLayer][batch+(i*batchSize)]=0.0;
			}
		}
	}
	
	iteratedOverLabels=true;
	if (iteratedOverImages){
		currentItem+=batchSize;
		if (currentItem>=totalItems-batchSize)
			currentItem=0;
		iteratedOverLabels=false;
		iteratedOverImages=false;
	}
	return labelsColumnMajor;
};
void BasicFileInput::initialise(RBM *currentRBM)
{
	// This is where pretty much everything gets initialised. 
	// However, we're not sure if anyone has already done this
	initialised=true;
	iteratedOverImages=false;
	iteratedOverLabels=false;
	
	int batchSize = currentRBM->batchSize;
	int inputSize = currentRBM->layerSizes[0] - currentRBM->labelSizes[0];
	int layers = currentRBM->numberOfNeuronLayers;
	int labelSize = currentRBM->labelSizes[currentRBM->numberOfNeuronLayers-1];

	setImagesFile(imagesFileName, totalItems, inputSize, batchSize);
	setLabelsFile(labelsFileName, totalItems, labelSize, layers, batchSize);

};
