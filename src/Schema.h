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
#ifndef SCHEMA_H
#define SCHEMA_H
#include "include/InputSource.h"
#include <vector>

struct SchemaItem{
	int size;
	InputSource *inputSource;
	bool onGPU;
	bool RBMLayer;
};

class Schema{
	public:
		std::vector< std::vector<SchemaItem> > layers;

		void addLayer(int RBMLayerSize);
		void addLayer(int RBMLayerSize, SchemaItem schemaItem);
		void addLayer(int RBMLayerSize, std::vector<SchemaItem> schemaItems);
		void addLayer(std::vector<SchemaItem> schemaItems);

		void addItem(int layer, InputSource *input);
		void addItem(int layer, SchemaItem);
};
#endif
