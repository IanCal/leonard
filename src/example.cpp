
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
