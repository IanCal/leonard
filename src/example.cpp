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

#include <rbm.cuh>
// which are needed
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include  <sys/timeb.h>

#include "SimpleController.h"
#include "BasicFileInput.h"
#include <allegro.h>
void drawImage(BITMAP *buffer, int xPos, int yPos, int scale, int width, int height, float *data, 
		int batchSize=1, int displayNumber=0, int spacing=0, int backgroundColour=0){
	
	int xCoord=0;
	int yCoord=0;
	int colour=0;
	if (backgroundColour)
		rectfill(buffer, xPos-spacing, yPos-spacing, xPos+(width*scale)+spacing, yPos+(height*scale)+spacing, makecol(backgroundColour,backgroundColour,backgroundColour));
	for( int i=0 ; i<width ; i++ )
	{
		for( int j=0 ; j<height ; j++ )
		{
			xCoord=xPos+(i*scale);
			yCoord=yPos+(j*scale);
			colour=(int)(data[batchSize*(j*width + i)+displayNumber]*255);
			rectfill(buffer, xCoord+spacing, yCoord+spacing, xCoord+scale-spacing, yCoord+scale-spacing, makecol(colour,colour,colour));
		}
		
	}
	
};	

int main(int argc, char *argv[])
{
	int layers=4;
	int layerSizes[4] = {784,128,128,128};
	int labelSizes[4] = {0,0,10,0};
	int fileSize=50000;
	int epochs=10;
	SimpleController* basicController = new SimpleController(0.005,fileSize,epochs);
	BasicFileInput*   basicInput = new BasicFileInput(argv[1],argv[2],fileSize);
	RBM *a = new RBM(layers,layerSizes,labelSizes,basicController,basicInput,32);
	
	printf("Created RBM\n");
	printf("Trying to train\n");
	
	
time_t start, end;
time(&start);
	for( int i=0 ; i<epochs*layers*(fileSize/32) ; i++ )
	{
		a->learningIteration();
	}
time(&end);
float timetaken=difftime(end,start);
printf("Took %f, running at a rate of %f/s\n",timetaken,float(fileSize*epochs)/timetaken);
	printf("rain\n");
	// testing of reading
	// Start allegro	
	if (allegro_init() != 0)
		return 1;

	install_mouse();
	install_keyboard();
	install_timer();
	//	show_os_cursor(1);
	enable_hardware_cursor();
	show_os_cursor(0);
	set_color_depth(24);
	if (set_gfx_mode(GFX_AUTODETECT_WINDOWED, 150*9, 150*5, 0, 0) != 0) {
		if (set_gfx_mode(GFX_SAFE, 150*9, 150*5, 0, 0) != 0) {
			set_gfx_mode(GFX_TEXT, 0, 0, 0, 0);
			allegro_message("Unable to set any graphic mode\n%s\n", allegro_error);
			return 1;
		}
	}
	BITMAP *buffer;

	float *current = new float[784*32];
	float *original = new float[784*32];
	float *labels = new float[10*32];
	a->setInputPattern();
	a->setLabels();
	a->pushUp(0, true, true, true);
	a->pushDown(0, false, true, true);
	a->getInput(0,current, true);
	a->getInput(0,original, false);
	a->getLabels(2,labels, true);
	printf("#labels %d\n",a->labelSizes[2]);
	buffer = create_bitmap(SCREEN_W, SCREEN_H);
	set_palette(desktop_palette);
	//Drawing loop
	while (!key[KEY_ESC]){
		if (key[KEY_DOWN]){
			//a->learningIteration();
		    a->setInputPattern();
		    a->setLabels();
			a->pushUp(0, true, true, true);
			a->pushUp(1, true, true, true);
			a->pushUp(2, true, true, true);
			a->pushDown(2, false, true, true);
			a->pushDown(1, false, true, true);
			a->pushDown(0, false, false, true);
			
			a->getInput(0,current,true);
			a->getInput(0,original,false);
			a->getLabels(2,labels, true);
			rest(10);
		}
		else{
			rest(10);
		}
		for( int x=0 ; x<8 ; x++ )
		{
			for( int y=0 ; y<4 ; y++ )
			{
				drawImage(buffer,5+150*x,10+150*y,2,28,28,original,32,x+y*8,0,15);	
				drawImage(buffer,60+150*x,10+150*y,3,28,28,current,32,x+y*8,0,15);	
				drawImage(buffer,10+150*x,130+150*y,8,10,1,labels,32,x+y*8,1,255);	
			}
			
		}
		
		blit(buffer, screen, 0, 0, 0, 0, SCREEN_W, SCREEN_H);
	}	
	return 0;
}
