/*
Farokh Confectioner
CS144 Neural network Project

This class contains all of the mathematical functions and activation function
used in this neural network.
*/
#pragma once
#include <math.h>
#include <iostream>

using namespace std;

class ActivationFunction {
public:

	//Calculates the result of the sigmoid function.
	static double SigmoidFunction(double x);
};