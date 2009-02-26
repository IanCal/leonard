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
#include "BasicFileInput.cuh"

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
	int layerSizes[4] = {10,10,10,10};
	int labelSizes[4] = {0,0,0,10};
	SimpleController* basicController = new SimpleController(0.01,1000,5);
	BasicFileInput*   basicInput = new BasicFileInput(argv[1],50000,28*28, 32);
	RBM *a = new RBM(4,layerSizes,labelSizes,basicController);

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

	buffer = create_bitmap(SCREEN_W, SCREEN_H);
	set_palette(desktop_palette);
float *current=basicInput->getNextInput(a,28*28,32);
	//Drawing loop
	while (!key[KEY_ESC]){
		if (key[KEY_DOWN]){
			current=basicInput->getNextInput(a,28*28,32);
			rest(10);
		}
		else{
			rest(10);
		}
		for( int x=0 ; x<8 ; x++ )
		{
			for( int y=0 ; y<4 ; y++ )
			{
				drawImage(buffer,10+150*x,10+150*y,4,28,28,current,32,x+y*8,1,15);	
			}
			
		}
		
		blit(buffer, screen, 0, 0, 0, 0, SCREEN_W, SCREEN_H);
	}	
	return 0;
}
