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
#include "Schema.h"


/** 
 * @brief Add a new layer to the schema, consisting of a RBM layer of size RBMLayerSize
 * 
 * @param RBMLayerSize The size of the RBM layer in neurons (does not include batch size)
 */
void Schema::addLayer(int RBMLayerSize){
	SchemaItem newItem;
	newItem.size = RBMLayerSize;
	newItem.RBMLayer = true;
	newItem.onGPU = true;
	newItem.inputSource = NULL;
	std::vector<SchemaItem> newLayer;
	newLayer.push_back(newItem);
	layers.push_back(newLayer);
};

/** 
 * @brief Adds a new layer to the schema, consisting of an RBM layer and another input source
 * 
 * @param RBMLayerSize The size of the RBM part of the layer (does not include batch size)
 * @param schemaItem This is a schema item, an input source of some kind.
 */
void Schema::addLayer(int RBMLayerSize, SchemaItem schemaItem){
	// First, create a new RBM layer
	addLayer(RBMLayerSize);
	(*layers.end()).push_back(schemaItem);
};



void Schema::addLayer(int RBMLayerSize, std::vector<SchemaItem> schemaItems){};
void Schema::addLayer(std::vector<SchemaItem> schemaItems){};
void Schema::addItem(int layer, InputSource *input){
	//Create new item, call addItem
};

void Schema::addItem(int layer, SchemaItem newItem){
	//Check that it's a valid layer choice
	if (layers.size() <= layer){
		layers[layer].push_back(newItem);
	}
	else{
		//Choice is invalid. Die?
	};
};
