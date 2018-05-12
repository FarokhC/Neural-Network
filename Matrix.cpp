/*
Farokh Confectioner
CS144 Neural network Project
*/

#include "Matrix.h"
#include "ActivationFunction.h"

 
//Throw this error when two matricies, which are not multipliable,
//are attempted to be multiplied.
class MultiplicationError : public std::exception {
public:
	const char * what() const throw()
	{
		return	"ERROR: The matricies are not compatible for multiplication!\n";
	}
};

 
Matrix::Matrix() {
}

 
Matrix::Matrix(int rows, int columns) {
	weights.resize(rows);
	for (int i = 0; i < rows; i++) {
		weights[i].resize(columns);
	}
}

 
void Matrix::SetVal(int row, int column, double value) {
	weights[row][column] = value;
}
 
 
double Matrix::GetVal(int row, int column) const {
	return weights[row][column];
}

 
Matrix Matrix::GetTranspose() {
	Matrix transpose_matrix(this->GetColumns(), this->GetRows());

	for (int row = 0; row < this->GetRows(); row++) {
		for (int column = 0; column < this->GetColumns(); column++) {
			transpose_matrix.SetVal(column, row, this->GetVal(row, column));
		}
	}
	return transpose_matrix;
}
 
 
Matrix Matrix::MultiplyMatricies(Matrix other) const {
	if (!this->IsMultipliable(other)) {
		MultiplicationError error;
		cout << error.what() << endl;
		throw error;
	}
	Matrix result_matrix(this->GetRows(), other.GetColumns());
	for (int i = 0; i < result_matrix.GetRows(); i++) {
		for (int j = 0; j < result_matrix.GetColumns(); j++) {
			double sum = 0;
			for (int k = 0; k < this->GetColumns(); k++) {
				sum += this->GetVal(i, k) * other.GetVal(k, j);
			}
			result_matrix.SetVal(i, j, sum);
		}
	}
	return result_matrix;
}

 
bool Matrix::IsMultipliable(Matrix other) const {
	return this->GetColumns() == other.GetRows();
}
 
 
int Matrix::GetRows() const {
	return weights.size();
}
 
 
int Matrix::GetColumns() const {
	const int FIRST_ARRAY_REF = 0;
	return weights[FIRST_ARRAY_REF].size();
}
 
 
void Matrix::Print() {
	for (int row = 0; row < this->GetRows(); row++) {
		for (int column = 0; column < this->GetColumns(); column++) {
			cout << weights[row][column] << " ";
		}
		cout << endl;
	}
	cout << endl;
}
 
Matrix Matrix::ApplySigmoid() const {
	Matrix new_matrix = Matrix(this->GetRows(), this->GetColumns());
	for (int i = 0; i < this->GetRows(); i++) {
		for (int j = 0; j < this->GetColumns(); j++) {
			double new_weight = ActivationFunction::SigmoidFunction(this->GetVal(i, j));
			new_matrix.SetVal(i, j, new_weight);
		}
	}
	return new_matrix;
}
 
 
vector<vector<double>> Matrix::GetWeightsVector() const {
	return weights;
}
 
 
Matrix Matrix::operator*(Matrix other) {
	return this->MultiplyMatricies(other);
}