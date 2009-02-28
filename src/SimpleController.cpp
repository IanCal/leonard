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

void SimpleController::updateParameters(RBM *currentRBM){
	
	if( currentSample>=samplesPerEpoch*trainingEpochs )
	{
		layerToTrain++;
		printf("Currently training layer %d\n",layerToTrain);
		if( layerToTrain>=currentRBM->numberOfWeightLayers )
		{
			learningRate=0;
		}
		for( int layer=0 ; layer<currentRBM->numberOfWeightLayers ; layer++ )
		{
			currentRBM->learningRates[layer]=0.;	
		}
		currentRBM->learningRates[layerToTrain]=learningRate;
		currentSample=0;
	}
	currentSample+=currentRBM->batchSize;
	
};

void SimpleController::initialise(RBM *currentRBM){
	for( int layer=0 ; layer<currentRBM->numberOfWeightLayers ; layer++ )
	{
		currentRBM->learningRates[layer]=0.;	
		currentRBM->momentum[layer]=0.;	
	}
	currentRBM->learningRates[0]=learningRate;
};
