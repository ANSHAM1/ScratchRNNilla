import ScratchRNNilla;
import Matrix;

#include <iostream>
#include <vector>
using namespace std;

int main() {
    auto [dataset, target] = Network::readStockSequenceFiles("Stock_Market_5y");
    auto [minVal, maxVal] = Network::normalizeDataset(dataset, target);

    int epochs = 10;
    double learningRate = 0.01;
    double decayRate = 0.05;
    double lembda = 0.01;
    double threshold = 0.5;

    int INPUT_SIZE = dataset[0][0].COLUMN_SIZE;
    int LAYER_SIZE = target[0].COLUMN_SIZE;

    Network model;
    model.RLayer(4, 4);
    model.Train(dataset, target, epochs, learningRate, decayRate, lembda, threshold);

    model.Test(dataset, target, minVal, maxVal);

    double RMSE = model.get_RMSE();
    double Rsqr = model.get_Rsqr();

    Matrix predicted = model.Predict(dataset[0], target[0]);
    Matrix Output = Network::denormalizeMatrix(predicted, minVal, maxVal);

    cout << "RMSE : " << RMSE << endl;
    cout << "R^2 Score : " << Rsqr << endl;

    using namespace std_Matrix;
    cout << Output;

    return 0;
}