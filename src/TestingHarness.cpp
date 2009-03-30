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
#include "TestingHarness.h"
#include <math.h>
#include <stdio.h>


TestingHarness::TestingHarness(InputSource *testInput){
	this->testingInput = testInput;
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

	RBMToTest->inputSource = testingInput;

	for( int i=0 ; i<iterations ; i++ )
	{
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
						if (i<2)
							printf("(%2.4f.%2.4f), ",initial[location],reconstruction[location]);
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
						if (i<2)
							printf("\n%d .. %d - %d XXX\n", batch+(i*batchSize), actual, bestv); 
					}
					else{
						if (i<2)
							printf("\n%d .. %d - %d\n", batch+(i*batchSize), actual, bestv); 
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
