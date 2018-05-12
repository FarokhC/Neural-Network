/*
Farokh Confectioner
CS144 Neural network Project

This class models a Matrix object, which contains rows and columns. 
*/

#pragma once
#include <vector>
#include <iostream>
#include "ActivationFunction.h"

using namespace std;

class Matrix {

public:

	//Constructs a matrix object.
	Matrix();
	
	//Constructs a matrix object with the number of rows
	//and columns specified.
	Matrix(int rows, int columns);

	//Sets value into the indexed row and column of the matrix.
	void SetVal(int row, int column, double value);

	//Returns the element at the specified row and column index.
	double GetVal(int row, int column) const;

	//Returns the transpose of a matrix.
	Matrix GetTranspose();

	//Multiplies this matrix with another matrix.
	//The columns of this matrix must equal the number of columns 
	//of the other matrix.
	Matrix MultiplyMatricies(Matrix other) const;

	//Determines whether two matricies are multipliable.
	bool IsMultipliable(Matrix other) const;

	//Returns the number of rows in a matrix.
	int GetRows() const;

	//Returns the number of columns in a matrix.
	int GetColumns() const;

	//Prints a matrix out.
	void Print();

	//Applies the sigmoid function to every element of the matrix and returns a matrix
	//with the updated values.
	Matrix ApplySigmoid() const;

	//Overloading of the * operator. 
	Matrix operator*(Matrix other);

private:

	//Returns the underlying weights vector.
	vector<vector<double>> GetWeightsVector() const;
	vector<vector<double>> weights;
};