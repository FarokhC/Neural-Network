/*
Farokh Confectioner
CS144 Neural network Project
*/

#include "ActivationFunction.h"

double ActivationFunction::SigmoidFunction(double x) {
	return 1 / (1 + exp(-x));
}