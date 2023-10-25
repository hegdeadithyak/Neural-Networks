#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
using namespace std;

float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}

int main()
{
    // Define the structure of the weights and biases
    int numInputs = 2;
    int numHidden = 2;
    int numOutputs = 1;
    

    float loadedHiddenWeights[numInputs][numHidden];
    float loadedHiddenBiases[numHidden];
    float loadedOutputWeights[numHidden][numOutputs];
    float loadedOutputBiases[numOutputs];
    

    // Load weights and biases from the text file
    std::ifstream inputFile("weights_and_biases.txt");
    if (inputFile.is_open())
    {
        for (int i = 0; i < numHidden; i++)
        {
            for (int j = 0; j < numInputs; j++)
            {
                inputFile >> loadedHiddenWeights[j][i];
            }
            inputFile >> loadedHiddenBiases[i];
        }
        for (int i = 0; i < numOutputs; i++)
        {
            for (int j = 0; j < numHidden; j++)
            {
                inputFile >> loadedOutputWeights[j][i];
            }
            inputFile >> loadedOutputBiases[i];
        }
        inputFile.close();
    }
    else
    {
        std::cerr << "Unable to open the input file.\n";
        return 1;
    }

    cout << "Hidden weights: " << endl;
    for (int i = 0; i < numHidden; i++)
    {
        for (int j = 0; j < numInputs; j++)
        {
            cout << loadedHiddenWeights[j][i] << " ";
        }
        cout << endl;
    }

    cout << "Hidden biases: " << endl;
    for (int i = 0; i < numHidden; i++)
    {
        cout << loadedHiddenBiases[i] << " ";
    }

    cout << endl;

    cout << "Output weights: " << endl;
    for (int i = 0; i < numOutputs; i++)
    {
        for (int j = 0; j < numHidden; j++)
        {
            cout << loadedOutputWeights[j][i] << " ";
        }
        cout << endl;
    }

    cout << "Output biases: " << endl;
    for (int i = 0; i < numOutputs; i++)
    {
        cout << loadedOutputBiases[i] << " ";
    }

    cout << endl;

    // Initialize new input data
    double new_input[numInputs] = {0.0, 1.0}; // Replace with your desired input values

    // Initialize arrays for the hidden layer and output layer
    double new_hiddenLayer[numHidden];
    double new_outputLayer[numOutputs];

    // Perform forward propagation
    for (int j = 0; j < numHidden; j++)
    {
        new_hiddenLayer[j] = 0.0;
        for (int k = 0; k < numInputs; k++)
        {
            new_hiddenLayer[j] += new_input[k] * loadedHiddenWeights[k][j];
        }
        new_hiddenLayer[j] += loadedHiddenBiases[j];
        new_hiddenLayer[j] = sigmoid(new_hiddenLayer[j]);
    }

    for (int j = 0; j < numOutputs; j++)
    {
        new_outputLayer[j] = 0.0;
        for (int k = 0; k < numHidden; k++)
        {
            new_outputLayer[j] += new_hiddenLayer[k] * loadedOutputWeights[k][j];
        }
        new_outputLayer[j] += loadedOutputBiases[j];
        new_outputLayer[j] = sigmoid(new_outputLayer[j]);
    }

    // The result is in new_outputLayer
    double prediction = new_outputLayer[0];


    if (prediction >0.9){
        prediction = 1;
    }
    else{
        prediction = 0;
    }
    // Print the prediction
    cout << "Prediction: " << prediction << endl;

    return 0;
}
