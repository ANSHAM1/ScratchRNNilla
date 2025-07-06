export module ScratchRNNilla;

import Matrix;
import ReverseAutoDiff;

import <iostream>;
import <vector>;
import <cmath>;
import <memory>;
import <fstream>;
import <sstream>;
import <filesystem>;
namespace fs = std::filesystem;
using namespace std;

class RNN {
public:
	vector<double> FEATURES;
	size_t Ht_SIZE;
	size_t Ht_1_SIZE;

	shared_ptr<Node> W;
	shared_ptr<Node> U;
	shared_ptr<Node> B;

	shared_ptr<Node> OUT;

	RNN(shared_ptr<Node>& w, shared_ptr<Node>& u, shared_ptr<Node>& b, size_t size, size_t sizePrev = 0)
		: W(w), U(u), B(b), Ht_SIZE(size), Ht_1_SIZE(sizePrev == 0 ? size : sizePrev) {}

	shared_ptr<Node> forward(shared_ptr<Node>& x, shared_ptr<Node>& ht_1) {
		OUT = tanh(x * W + ht_1 * U + B);
		return OUT;
	}
};

export class Network {
	RNN* TS;

	shared_ptr<Node> Wh;
	shared_ptr<Node> Uh;
	shared_ptr<Node> Bh;

	shared_ptr<Node> Wo;
	shared_ptr<Node> Bo;

	double RMSE = 0;
	double R2_SCORE = 0;

public:
	Network() {}

	void RLayer(size_t size, size_t sizePrev = 0) {
		Wh = make_shared<Node>(Matrix(size, sizePrev, "random"));
		Uh = make_shared<Node>(Matrix(sizePrev, sizePrev, "random"));
		Bh = make_shared<Node>(Matrix(1, sizePrev, 0));
		
		Wo = make_shared<Node>(Matrix(sizePrev, sizePrev, "random"));
		Bo = make_shared<Node>(Matrix(1, sizePrev, 0));

		TS = new RNN(Wh, Uh, Bh, size, sizePrev);
	}

	void Train(vector<vector<Matrix>>& dataset, vector<Matrix>& Targets, int epochs, double learningRate, double decayRate, double lembda, double threshold = 0.5) {
		for (int i = 0; i < epochs; i++) {
			double Loss = 0;
			for (int j = 0; j < dataset.size(); j++) {
				Matrix Target = Targets[j];

				shared_ptr<Node> ht_1 = make_shared<Node>(Matrix(1, TS->Ht_1_SIZE, 0));
				for (size_t k = 0; k < dataset[j].size(); k++) {
					shared_ptr<Node> x = make_shared<Node>(dataset[j][k]);
					ht_1 = TS->forward(x, ht_1);
				}

				shared_ptr<Node> Oo = sigmoid(ht_1 * Wo + Bo);
				Loss += MSE(Oo, Target);

				clippedGrad(Wh, Uh, Wo, threshold);

				Wo->DATA = Wo->DATA - (learningRate * (Wo->GRADIENT + lembda * Wo->DATA));
				Bo->DATA = Bo->DATA - (learningRate * Bo->GRADIENT);

				Wh->DATA = Wh->DATA - (learningRate * (Wh->GRADIENT + lembda * Wh->DATA));
				Uh->DATA = Uh->DATA - (learningRate * (Uh->GRADIENT + lembda * Uh->DATA));
				Bh->DATA = Bh->DATA - (learningRate * Bh->GRADIENT);


				Wo->GRADIENT = Matrix(Wo->DATA.ROW_SIZE, Wo->DATA.COLUMN_SIZE, 0);
				Bo->GRADIENT = Matrix(Bo->DATA.ROW_SIZE, Bo->DATA.COLUMN_SIZE, 0);

				Wh->GRADIENT = Matrix(Wh->DATA.ROW_SIZE, Wh->DATA.COLUMN_SIZE, 0);
				Uh->GRADIENT = Matrix(Uh->DATA.ROW_SIZE, Uh->DATA.COLUMN_SIZE, 0);
				Bh->GRADIENT = Matrix(Bh->DATA.ROW_SIZE, Bh->DATA.COLUMN_SIZE, 0);

				showProgressBar(j + 1, dataset.size());
			}
			std::cout << endl;
			learningRate = learningRate / (1 + decayRate * i);
			cout << "Loss per epoch : " << Loss / dataset.size() << endl;
		}
	}

	void Test(vector<vector<Matrix>>& dataset, vector<Matrix>& Targets , const vector<double>& minVal, const vector<double>& maxVal) {
		vector<Matrix> predicted;
		for (int data = 0; data < dataset.size(); data++) {
			shared_ptr<Node> ht_1 = make_shared<Node>(Matrix(1, TS->Ht_1_SIZE, 0));
			for (size_t i = 0; i < dataset[data].size(); i++) {
				shared_ptr<Node> x = make_shared<Node>(dataset[data][i]);
				ht_1 = TS->forward(x, ht_1);
			}

			shared_ptr<Node> Oo = sigmoid(ht_1 * Wo + Bo);
			predicted.push_back(Oo->DATA);
			RMSE += std_Loss::MSE(Oo->DATA, Targets[data]);
		}
		RMSE /= dataset.size();
		RMSE = pow(RMSE, 0.5);

		rsqr(predicted, Targets, minVal, maxVal);
	}

	Matrix Predict(vector<Matrix>& dataset, Matrix& target) {
		shared_ptr<Node> ht_1 = make_shared<Node>(Matrix(1, TS->Ht_1_SIZE, 0));
		for (size_t i = 0; i < dataset.size(); i++) {
			shared_ptr<Node> x = make_shared<Node>(dataset[i]);
			ht_1 = TS->forward(x, ht_1);
		}

		shared_ptr<Node> Oo = sigmoid(ht_1 * Wo + Bo);
		return Oo->DATA;
	}

	double get_RMSE() const {
		return RMSE;
	}

	double get_Rsqr() const {
		return R2_SCORE;
	}

	static pair<vector<vector<Matrix>>, vector<Matrix>> readStockSequenceFiles(const string& folderPath) {
		vector<vector<Matrix>> Dataset;
		vector<Matrix> Target;
		cout << "Reading Files :   ";
		for (const auto& entry : fs::directory_iterator(folderPath)) {
			if (!entry.is_regular_file() || entry.path().extension() != ".csv")
				continue;

			ifstream file(entry.path());
			if (!file.is_open()) continue;

			string line;
			vector<Matrix> Sample;

			// Skip header
			getline(file, line);
			int count = 0;
			while (getline(file, line) && count < 100) {
				stringstream ss(line);
				string token;
				Matrix Features = Matrix(1, 4, 0);

				// Skip date
				getline(ss, token, ',');

				// Read open, high, low, close
				bool valid = true;
				for (int i = 0; i < 4; ++i) {
					if (!getline(ss, token, ',')) {
						valid = false;
						break;
					}
					try {
						Features.set(0, i, stod(token));
					}
					catch (...) {
						valid = false;
						break;
					}
				}

				if (!valid || Features.isZeroMatrix())
					continue;

				Sample.push_back(Features);
				++count;
			}

			file.close();

			size_t size = Sample.size();
			Target.push_back(Sample[size - 1]);
			Sample.pop_back();
			Dataset.push_back(Sample);

		}
		cout << " -> Completed" << endl;
		return { Dataset, Target };
	}

	static pair<vector<double>, vector<double>> normalizeDataset(vector<vector<Matrix>>& Dataset, vector<Matrix>& Target) {
		const int FEATURE_SIZE = 4;

		// Initialize min and max for each feature
		vector<double> minVal(FEATURE_SIZE, DBL_MAX);
		vector<double> maxVal(FEATURE_SIZE, -DBL_MAX);

		// Pass 1: Find min and max across all samples
		for (const auto& sample : Dataset) {
			for (const auto& mat : sample) {
				for (int i = 0; i < FEATURE_SIZE; ++i) {
					double val = mat.get(0, i);
					minVal[i] = min(minVal[i], val);
					maxVal[i] = max(maxVal[i], val);
				}
			}
		}

		// Also include target in min/max calc
		for (const auto& mat : Target) {
			for (int i = 0; i < FEATURE_SIZE; ++i) {
				double val = mat.get(0, i);
				minVal[i] = min(minVal[i], val);
				maxVal[i] = max(maxVal[i], val);
			}
		}

		// Pass 2: Normalize Dataset to [0, 1]
		for (auto& sample : Dataset) {
			for (auto& mat : sample) {
				for (int i = 0; i < FEATURE_SIZE; ++i) {
					double val = mat.get(0, i);
					if (maxVal[i] != minVal[i])
						mat.set(0, i, (val - minVal[i]) / (maxVal[i] - minVal[i]));
					else
						mat.set(0, i, 0.0); // avoid division by zero
				}
			}
		}

		// Normalize Targets to [0, 1]
		for (auto& mat : Target) {
			for (int i = 0; i < FEATURE_SIZE; ++i) {
				double val = mat.get(0, i);
				if (maxVal[i] != minVal[i])
					mat.set(0, i, (val - minVal[i]) / (maxVal[i] - minVal[i]));
				else
					mat.set(0, i, 0.0);
			}
		}

		return { minVal, maxVal };
	}

	static Matrix denormalizeMatrix(const Matrix& normalized, const vector<double>& minVal, const vector<double>& maxVal) {
		Matrix result = normalized;
		for (int i = 0; i < 4; ++i) {
			if (maxVal[i] != minVal[i]) {
				double val = normalized.get(0, i);
				double denorm = val * (maxVal[i] - minVal[i]) + minVal[i];
				result.set(0, i, denorm);
			}
			else {
				result.set(0, i, minVal[i]);
			}
		}
		return result;
	}

private:
	void clippedGrad(shared_ptr<Node>& Wh, shared_ptr<Node>& Bh, shared_ptr<Node>& Wo, double threshold = 0.5) {
		double global_norm = sqrt(squaredSum(Wh->DATA) + squaredSum(Bh->DATA) + squaredSum(Wo->DATA));

		if (global_norm > threshold) {
			double scale = threshold / global_norm;

			Wh->DATA = Wh->DATA * scale;
			Uh->DATA = Uh->DATA * scale;
			Wo->DATA = Wo->DATA * scale;
		}
	}

	double squaredSum(Matrix& M) const {
		double sum = 0.0;
		for (int i = 0; i < M.ROW_SIZE; ++i)
			for (int j = 0; j < M.COLUMN_SIZE; ++j)
				sum += M.get(i, j) * M.get(i, j);
		return sum;
	}

	double mean(vector<double>& flatvector) {
		double sum = 0;
		for (int i = 0; i < flatvector.size(); i++) {
			sum += flatvector[i];
		}
		return sum / flatvector.size();
	}

	vector<double> flatten(vector<Matrix>& data) {
		vector<double> flatvector;
		for (int d = 0; d < data.size(); d++) {
			for (int i = 0; i < data[d].ROW_SIZE; i++) {
				for (int j = 0; j < data[d].COLUMN_SIZE; j++) {
					flatvector.push_back(data[d].get(i, j));
				}
			}
		}
		return flatvector;
	}

	void rsqr(vector<Matrix>& predicted, vector<Matrix>& targets, const vector<double>& minVal, const vector<double>& maxVal) {
		vector<Matrix> P;
		vector<Matrix> T;
		for (int i = 0; i < predicted.size(); i++) {
			P.push_back(denormalizeMatrix(predicted[i], minVal, maxVal));
			T.push_back(denormalizeMatrix(targets[i], minVal, maxVal));
		}

		vector<double> p = flatten(P);
		vector<double> t = flatten(T);

		double mt = mean(t);

		double num = 0;
		double den = 0;
		for (int i = 0; i < predicted.size(); i++) {
			num += pow(t[i] - p[i], 2);
			den += pow(t[i] - mt, 2);
		}

		R2_SCORE = 1 - (num / den);
	}
};