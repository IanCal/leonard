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
#include "include/ParameterController.h"

/**
 * Constructor for a simple parameter controller.
 * @param iterationLearningRate The learning rate to be used. 
 * @param samples The number of samples per epoch.
 * @param epochs The number of epochs to train each layer for.
 */
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

