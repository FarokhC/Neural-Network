/*
Farokh Confectioner
CS144 Neural network Project

This class performs calculations to train a neural network using matricies.
It stores the calculated weights and allows a user to use the calculated 
weights to determine the value of a 28 x 28 pixel input image. The image 
must be in black and white, with colors ranging from 0 - 255.
*/

#pragma once

#include "Matrix.h"
#include "ActivationFunction.h"
#include <fstream>
#include <string>

using namespace std;

class NeuralNetwork {
public:

	const int INPUT_DIMENSIONS = 28;
	const int INPUT_NODES = 784;
	const int HIDDEN_NODES = 100;
	const int OUTPUT_NODES = 10;
	const double LEARNING_RATE = 0.3;

	//Initializes the NeuralNetwork object with input nodes, 
	//hidden nodes, output nodes, and learning rate.
	void Initialization(int input_nodes, int hidden_nodes, 
		int output_nodes, double learning_rate);

	//Trains the NeuralNetwork and updates the weights.
	void train();

	//Calculates and outputs the likelyhood of the input number.
	void query();

private:
	int input_nodes;
	int hidden_nodes;
	int output_nodes;
	double learning_rate;

	Matrix w_input_hidden;
	Matrix w_hidden_output;

	//Generates a matrix with random weights, ranging from -1 / sqrt(y_size) 
	//to 1 / sqrt(y_size)
	Matrix RandomWeightsMatrix(int x_size, int y_size);

	//Calculatees the final error at the end of the output layer.
	Matrix CaclulateFinalError(Matrix calc_output, 
		Matrix expected_output);
	
	//Calculates error based off training data and output data.
	double CalculateError(double training_data, double output_data);

	//Calculates the Slope Error
	Matrix SlopeError(Matrix weights, Matrix output, 
		Matrix training_data);

	//Updates the weights of the matrix with the new weights, 
	//and returns these new weights.
	Matrix UpdateWeights(Matrix old_weights, double learning_rate, 
		Matrix slope_error);

	//Returns the expected output matrix.
	Matrix GetExpectedOutputs(int expected_output);
	
	//Gets the number value of the output matrix.
	int GetNumberValue(Matrix output_matrix);
};