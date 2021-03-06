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
#include "TestingHarness.h"
#include "include/InputSource.h"
#include <math.h>
#include <stdio.h>


TestingHarness::TestingHarness(InputSource *testInput, ParameterController *parameterController){
	this->testingInput = testInput;
    this->parameterUpdater = parameterController;
};

float TestingHarness::train(RBM *RBMToTest, int iterations){

    parameterUpdater->initialise(RBMToTest);

    for (int i = 0; i < iterations / (RBMToTest->batchSize) ; i++) {
        testingInput->getNextInput(RBMToTest);
        testingInput->getNextLabel(RBMToTest);
        RBMToTest->updateWeights();
        parameterUpdater->updateParameters(RBMToTest);
    }

};

float TestingHarness::test(RBM *RBMToTest, int iterations){
	
	float mse=0.0;
	float errorProportion=0.0;
	int maxItem, actual, bestv;
	int location;
	float maxValue;
	int batchSize = RBMToTest->batchSize;
	iterations /= batchSize;
	float initial[batchSize * testingInput->maxLabels];
	float reconstruction[batchSize * testingInput->maxLabels];

	for( int i=0 ; i<iterations ; i++ )
	{
		testingInput->getNextInput(RBMToTest);
        testingInput->getNextLabel(RBMToTest);
        RBMToTest->classify();
		for( int layer=0 ; layer<RBMToTest->numberOfNeuronLayers ; layer++ )
		{
			if( RBMToTest->labelSizes[layer]!=0 )
			{
				//get initial and reconstruction
				RBMToTest->getLabels(layer,initial,false);
				RBMToTest->getLabels(layer,reconstruction,true);
				// Look at individual classifications
				for( int batch=0 ; batch<batchSize ; batch++ )
				{	
					maxValue=-1.0;
					maxItem=0;
					actual=0;
					bestv=0;
					for( int pos=0 ; pos<(RBMToTest->labelSizes[layer]) ; pos++ )
					{
						location=batch+(pos*batchSize);
						//if (i<2)
						//	printf("(%2.4f.%2.4f), ",initial[location],reconstruction[location]);
						if (initial[location]>0.0)
							actual=pos;
						if (reconstruction[location]>maxValue)
						{
							maxValue = reconstruction[location];
							maxItem = location;
							bestv=pos;
						}
					}
					if (initial[maxItem]<0.5f){
						errorProportion+=1.;
						//if (i<2)
						//	printf("\n%d .. %d - %d XXX\n", batch+(i*batchSize), actual, bestv); 
					}
					else{
						//if (i<2)
						//	printf("\n%d .. %d - %d\n", batch+(i*batchSize), actual, bestv); 
					}
				}
				//printf("\n");
				// Look at the MSE	
				for( int pos=0 ; pos<RBMToTest->labelSizes[layer] * batchSize; pos++ )
				{
					mse+=pow(initial[pos]-reconstruction[pos],2);
				}
			}
		}
	}
	printf("Total errors: %f\n",errorProportion);
	mse /= batchSize*iterations;
	errorProportion /= batchSize*iterations;
	return errorProportion;
};
