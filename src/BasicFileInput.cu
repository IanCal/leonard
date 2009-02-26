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
#include "BasicFileInput.cuh"

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
#include <time.h>
#include  <sys/timeb.h>
#include <allegro.h>

BasicFileInput::BasicFileInput(char *imagesFileName, char *labelsFileName, int length);

{
	//Set the basic variables
	currentItem = 0;
	totalItems = length;
	initialised = false;

}

void BasicFileInput::setImagesFile(char *filename, int length, int itemSize, int batchSize)
{

	//Need to initialise the array that will store the image
	inputColumnMajor = (float*) malloc(batchSize*itemLength * sizeof(float));	
	if (!inputColumnMajor){
		printf("Could not allocate array of size %dMB for input file, dying\n",(length * sizeof(float))/1e6);	
		exit(EXIT_FAILURE);
	}

	inputFile = open(inputName,O_RDONLY);
	if (inputFile == -1){
		printf("Error opening the file %s", inputName);
		exit(EXIT_FAILURE);
	}

	//Now mmap it so we don't need the whole file in memory at one time

	inputFileMap = (float *) mmap(0, fileSize*sizeof(float), PROT_READ, MAP_SHARED, inputFile, 0);
	if (inputFileMap == MAP_FAILED) {
		close(inputFile);
		printf("Error mmapping the file %s", inputName);
		exit(EXIT_FAILURE);
	}
};


float* BasicFileInput::getNextInput(RBM *currentRBM)
{
	int batchSize = RBM->batchSize;
	int inputSize = RBM->layerSizes[0]-RBM->labelSizes[0];
	for( int batch=0 ; batch<batchSize ; batch++ )
	{
		for( int i=0 ; i<inputSize ; i++ )
		{
			inputColumnMajor[batch+(i*batchSize)]=inputFileMap[(currentPosition+i)+inputSize*batch];
		}
	}
	currentPosition+=inputSize*batchSize;
	if (currentPosition>=fileSize-(inputSize*batchSize))
		currentPosition=0;
	return inputColumnMajor;
};

float** BasicFileInput::getNextLabel(RBM *currentRBM)
{

	// Set up the labels...
};
void BasicFileInput::initialise(RBM *currentRBM)
{
	// This is where pretty much everything gets initialised. 
	// However, we're not sure if anyone has already done this
	

	totalItems=length;

};
