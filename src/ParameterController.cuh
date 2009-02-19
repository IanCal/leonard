#ifndef PARAMETERCONTROLLER
#define PARAMETERCONTROLLER
#include "rbm.cuh"
class ParameterController{
	public:
		ParameterController(){};
		virtual void updateParameters(RBM *currentRBM){};
		virtual void initialise(RBM *currentRBM){};
};
#endif
