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
#ifndef INPUTSOURCE
#define INPUTSOURCE
#include "rbm.cuh"
class InputSource{
	public:
		int maxLabels;
		InputSource(){};
		virtual float* getNextInput(RBM *currentRBM){return 0;};
		virtual float** getNextLabel(RBM *currentRBM){return 0;};
		virtual void initialise(RBM *currentRBM){maxLabels=0;};
};
#endif
