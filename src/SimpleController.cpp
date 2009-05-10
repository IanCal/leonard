/*
 *   This file is part of Leonard.
 *
 *   Leonard is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Leonard is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Leonard.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "SimpleController.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

SimpleController::SimpleController(float iterationLearningRate, int samples, int epochs){
	learningRate = iterationLearningRate;
	samplesPerEpoch = samples;
	trainingEpochs = epochs;
	layerToTrain=0;
	currentSample=0;
};

/**
 * This function is called every RBM learning iteration.
 * @param currentRBM This is a pointer to the RBM
 */

void SimpleController::updateParameters(RBM *currentRBM){
	
	if( currentSample>=samplesPerEpoch*trainingEpochs )
	{
		layerToTrain++;
		if( layerToTrain>=currentRBM->numberOfWeightLayers )
		{
			learningRate=0;
		}
		for( int layer=0 ; layer<currentRBM->numberOfWeightLayers ; layer++ )
		{
			currentRBM->learningRates[layer]=0.;	
			currentRBM->biasLearningRates[layer]=0.;
		}
		if (layerToTrain<currentRBM->numberOfWeightLayers)
		{
			printf("Currently training layer %d with learning rate %f\n",layerToTrain, learningRate);
			currentRBM->learningRates[layerToTrain]=learningRate;
			currentRBM->biasLearningRates[layerToTrain]=learningRate*0.05;
			currentRBM->weightDecay[layerToTrain]=0.999;
		}
		else 
		{
			printf("Overshoot\n");
		};
		currentSample=0;
	}
	/*
	if (currentSample<=samplesPerEpoch*3)
		currentRBM->learningRates[0]=0.0;
	else{
		currentRBM->learningRates[0]=learningRate;
	}
	*/
	currentSample+=currentRBM->batchSize;
	
};

void SimpleController::initialise(RBM *currentRBM){
	for( int layer=0 ; layer<currentRBM->numberOfWeightLayers ; layer++ )
	{
		currentRBM->learningRates[layer]=0.;	
		currentRBM->momentum[layer]=0.;	
	}
	currentRBM->learningRates[0]=learningRate;
	currentRBM->weightDecay[0]=0.999;
};
