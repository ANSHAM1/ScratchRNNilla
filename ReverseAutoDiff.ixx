export module ReverseAutoDiff;

import Matrix;

import <vector>;
import <memory>;
import <functional>;
using namespace std;

export class Node {
public:
	Matrix DATA;
	Matrix GRADIENT;

	vector<shared_ptr<Node>> PARENTS;
	function<void()> backward;

	Node(const Matrix& data) :
		DATA(data),
		GRADIENT(Matrix(data.ROW_SIZE, data.COLUMN_SIZE, 0)),
		backward([]() {}) {
	}
};

export shared_ptr<Node> operator*(const shared_ptr<Node>& A, const shared_ptr<Node>& B) {
	Matrix M = A->DATA * B->DATA;

	auto result = std::make_shared<Node>(M);
	result->PARENTS = { A, B };

	result->backward = [A, B, result]() {
		// dL/dA = dL/dC * dC/dA = GRADIENT * B.transpose
		A->GRADIENT = A->GRADIENT + result->GRADIENT * B->DATA.Transpose();

		// dL/dB = dL/dC * dC/dB = A.transpose * GRADIENT 
		B->GRADIENT = B->GRADIENT + A->DATA.Transpose() * result->GRADIENT;

		A->backward();
		B->backward();
		};

	return result;
}

export shared_ptr<Node> operator+(const shared_ptr<Node>& A, const shared_ptr<Node>& B) {
	Matrix M = A->DATA + B->DATA;

	auto result = std::make_shared<Node>(M);
	result->PARENTS = { A, B };

	result->backward = [A, B, result]() {
		// dL/dA = dL/dC  = GRADIENT
		A->GRADIENT = A->GRADIENT + result->GRADIENT;

		// dL/dB = dL/dC = GRADIENT 
		B->GRADIENT = B->GRADIENT + result->GRADIENT;

		A->backward();
		B->backward();
		};

	return result;
}

export shared_ptr<Node> operator-(const shared_ptr<Node>& A, const shared_ptr<Node>& B) {
	Matrix M = A->DATA - B->DATA;

	auto result = std::make_shared<Node>(M);
	result->PARENTS = { A, B };

	result->backward = [A, B, result]() {
		// dL/dA = dL/dC  = GRADIENT
		A->GRADIENT = A->GRADIENT + result->GRADIENT;

		// dL/dB = dL/dC = -GRADIENT 
		B->GRADIENT = B->GRADIENT - result->GRADIENT;

		A->backward();
		B->backward();
		};

	return result;
}

export shared_ptr<Node> elementaryProduct(const shared_ptr<Node>& A, const shared_ptr<Node>& B) {
	Matrix M = std_Matrix::elementaryProduct(A->DATA, B->DATA);

	auto result = std::make_shared<Node>(M);
	result->PARENTS = { A, B };

	result->backward = [A, B, result]() {
		// dL/dA = dL/dC  = GRADIENT ⊙ B
		A->GRADIENT = A->GRADIENT + std_Matrix::elementaryProduct(result->GRADIENT, B->DATA);

		// dL/dB = dL/dC = GRADIENT ⊙ A
		B->GRADIENT = B->GRADIENT + std_Matrix::elementaryProduct(result->GRADIENT, A->DATA);

		A->backward();
		B->backward();
		};

	return result;
}

export shared_ptr<Node> relu(const shared_ptr<Node>& A) {
	Matrix M = A->DATA.reluMat();

	auto result = std::make_shared<Node>(M);
	result->PARENTS = { A };

	result->backward = [A, result]() {
		// dL/dA = dL/dC * dC/dA = GRADIENT ⊙ dRelu
		A->GRADIENT = A->GRADIENT + std_Matrix::elementaryProduct(result->DATA.dReluMat(), result->GRADIENT);

		A->backward();
		};

	return result;
}

export shared_ptr<Node> tanh(const shared_ptr<Node>& A) {
	Matrix M = A->DATA.tanhMat();

	auto result = std::make_shared<Node>(M);
	result->PARENTS = { A };

	result->backward = [A, result]() {
		// dL/dA = dL/dC * dC/dA = GRADIENT ⊙ dTanh
		A->GRADIENT = A->GRADIENT + std_Matrix::elementaryProduct(result->DATA.dTanhMat(), result->GRADIENT);

		A->backward();
		};

	return result;
}

export shared_ptr<Node> sigmoid(const shared_ptr<Node>& A) {
	Matrix M = A->DATA.sigmoidMat();

	auto result = std::make_shared<Node>(M);
	result->PARENTS = { A };

	result->backward = [A, result]() {
		// dL/dA = dL/dC * dC/dA = GRADIENT ⊙ dSigmoid
		A->GRADIENT = A->GRADIENT + std_Matrix::elementaryProduct(result->DATA.dSigmoidMat(), result->GRADIENT);

		A->backward();
		};

	return result;
}

export pair<Matrix, double> Softmaxed_CCE(const shared_ptr<Node>& A, const Matrix& target) {
	Matrix softmaxed = softmax(A->DATA);
	double Loss = std_Loss::CCE(softmaxed, target);

	A->GRADIENT = std_Loss::dSoft_CCE(softmaxed, target);
	A->backward();

	return { softmaxed, Loss };
}

export double CCE(const shared_ptr<Node>& A, const Matrix& target) {
	double Loss = std_Loss::CCE(A->DATA, target);

	A->GRADIENT = std_Loss::dCCE(A->DATA, target);
	A->backward();

	return Loss;
}

export double BCE(const shared_ptr<Node>& A, const Matrix& target) {
	double Loss = std_Loss::BCE(A->DATA, target);

	A->GRADIENT = std_Loss::dBCE(A->DATA, target);
	A->backward();

	return Loss;
}

export double MSE(const shared_ptr<Node>& A, const Matrix& target) {
	double Loss = std_Loss::MSE(A->DATA, target);

	A->GRADIENT = std_Loss::dMSE(A->DATA, target);
	A->backward();

	return Loss;
}