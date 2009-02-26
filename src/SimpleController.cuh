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
#include "ParameterController.cuh"

class SimpleController: public ParameterController{
	public:
		float learningRate;
		int samplesPerEpoch;
		int trainingEpochs;
		int currentSample;
		int layerToTrain;
		SimpleController(float iterationLearningRate, int samples, int epochs);
		void updateParameters(RBM *currentRBM);
		void initialise(RBM *currentRBM);
};
