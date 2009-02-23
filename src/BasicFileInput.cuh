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
#ifndef BASICFILEINPUT
#define BASICFILEINPUT
#include "rbm.cuh"
#include "InputSource.cuh"

class BasicFileInput: public InputSource{
	public:
		int currentPosition;
		int fileSize;
		float *inputFileMap;
		float *inputColumnMajor;
		int inputFile;
		char *inputFileName;

		BasicFileInput(char *inputName, int length, int itemlength, int batchSize);
		float* getNextInput(RBM *currentRBM, int inputSize, int batchSize);
		void initialise(RBM *currentRBM);
};
#endif
