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

#include "rbm.cuh"
#include "SimpleController.cuh"

int main(int argc, char *argv[])
{
	int layerSizes[4] = {10,10,10,10};
	int labelSizes[4] = {0,0,0,10};
	SimpleController* basicController = new SimpleController(0.01,1000,5);
	RBM *a = new RBM(4,layerSizes,labelSizes,basicController);
	
	return 0;
}
