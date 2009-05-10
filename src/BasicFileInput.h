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
#ifndef BASICFILEINPUT
#define BASICFILEINPUT
#include "include/rbm.cuh"
#include "include/InputSource.h"

class BasicFileInput: public InputSource{
	public:
		int currentItem;
		int totalItems;
		bool initialised;
		
		char *imagesFileName;
		char *labelsFileName;

		float *imagesFileMap;
		int imagesFile;
		int inputFile;

		unsigned char *allLabels;
		float *inputColumnMajor;
		float **labelsColumnMajor;

		bool iteratedOverImages;
		bool iteratedOverLabels;

		BasicFileInput(char *imagesFileName, char *labelsFileName, int length);


		void setImagesFile(char *filename, int length, int itemSize, int batchSize);
		void setLabelsFile(char *filename, int fileLength, int labelSize, int layers, int batchSize);
		
		float* getNextInput(RBM *currentRBM);
		float** getNextLabel(RBM *currentRBM);

		void initialise(RBM *currentRBM);
};
#endif
