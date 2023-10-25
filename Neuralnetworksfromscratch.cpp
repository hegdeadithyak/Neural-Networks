#include <iostream>
#include <math.h> 
#include <vector>
#include <random>
#include <fstream>

using namespace std;


// Sigmoid function 
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

// Derivative of sigmoid function
double sigmoid_derivative(double x){
    return x * (1 - x);
}

//initialize weights and biases
double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }


//Shuffle the dataset
void shuffle(int *dataset, int size){
    for (int i = 0; i < size; i++){
        int random_index = rand() % size;
        swap(dataset[i], dataset[random_index]);
    }
}

#define numInputs 2
#define numHidden 2
#define numOutputs 1
#define numTrainingSets 4

int main(){

    //defining learning rate
    const double lr=0.1f;

    double hiddenLayer[numHidden];
    double outputLayer[numOutputs];
    
    double hiddenLayerBias[numHidden];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHidden];
    double outputWeights[numHidden][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = {{0.0f,0.0f},{0.0f,1.0f},{1.0f,0.0f},{1.0f,1.0f}};
    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},{1.0f},{1.0f},{1.0f}};

    for(int i=0;i<numInputs;i++){
        for(int j=0;j<numHidden;j++){
            hiddenWeights[i][j]=init_weight();
        }
    }

    for(int i=0;i<numHidden;i++){
        for(int j=0;j<numOutputs;j++){
            outputWeights[i][j]=init_weight();
        }
    }

    for(int i=0;i<numHidden;i++){
        hiddenLayerBias[i]=init_weight();
    }

    int trainingSetOrder[numTrainingSets]={0,1,2,3};

    int epochs = 10000;

    for(int numepochs=0;numepochs<epochs;numepochs++){
        
        
        shuffle(trainingSetOrder,numTrainingSets);

        //forward propagation
        for(int i=0;i<numTrainingSets;i++){


            int index=trainingSetOrder[i];
            
            
            for(int j=0;j<numHidden;j++){
                hiddenLayer[j]=0.0f;
                for(int k=0;k<numInputs;k++){

                    hiddenLayer[j]+=training_inputs[index][k]*hiddenWeights[k][j]; //Mathematically : W*X
                
                }
                
                hiddenLayer[j]+=hiddenLayerBias[j]; // Mathematically : sumof(W*X) + B
                
                
                hiddenLayer[j]=sigmoid(hiddenLayer[j]);// Mathematically : sigmoid(sumof(W*X) + B) ie adjusting it in the range of 0 to 1
            }

            for(int j=0;j<numOutputs;j++){
                
                outputLayer[j]=0.0f;
                
                for(int k=0;k<numHidden;k++){
                
                    outputLayer[j]+=hiddenLayer[k]*outputWeights[k][j];//Mathematically : outputLayer = sigmoid(sumof(W*X) + B) * Output weights
                
                }

                outputLayer[j] += outputLayerBias[j]; // Mathematically : outputLayer = sigmoid(sumof(W*X) + B) * Output weights + B

                outputLayer[j] = sigmoid(outputLayer[j]); // Mathematically : outputLayer = sigmoid(sigmoid(sumof(W*X) + B) * Output weights + B)
            }

            cout << "Input: " << training_inputs[index][0] << " " << training_inputs[index][1] << endl;
            cout << "Expected Output: " << training_outputs[index][0] << endl;
            cout << "Actual Output: " << outputLayer[0] << endl;
            cout <<"Epoch: "<< numepochs << endl;

            //Backpropagation starts here
            
            double outputLayerError[numOutputs];
            for(int j=0;j<numOutputs;j++){

                outputLayerError[j]=training_outputs[index][j]-outputLayer[j];   //Mathematically : True output - Predicted output
            }

            double outputDelta[numOutputs];
            for(int j=0;j<numOutputs;j++){
                outputDelta[j]=outputLayerError[j]*sigmoid_derivative(outputLayer[j]); //Mathematically : (True output - Predicted output) * sigmoid_derivative(outputLayer[j])
            }

            double hiddenLayerError[numHidden];
            for(int j=0;j<numHidden;j++){
                hiddenLayerError[j]=0.0f;
                for(int k=0;k<numOutputs;k++){
                    hiddenLayerError[j]+=outputDelta[k]*outputWeights[j][k]; //Mathematically : (True output - Predicted output) * sigmoid_derivative(outputLayer[j]) * outputWeights[j][k]
                }
            }

            double hiddenDelta[numHidden];
            for(int j=0;j<numHidden;j++){
                hiddenDelta[j]=hiddenLayerError[j]*sigmoid_derivative(hiddenLayer[j]); //Mathematically : (True output - Predicted output) * sigmoid_derivative(outputLayer[j]) * outputWeights[j][k] * sigmoid_derivative(hiddenLayer[j])
            }

            for(int j=0;j<numOutputs;j++){
                for(int k=0;k<numHidden;k++){
                    outputWeights[k][j]+=lr*hiddenLayer[k]*outputDelta[j]; //Mathematically : output weights += (True output - Predicted output) * sigmoid_derivative(outputLayer[j]) * outputWeights[j][k] * sigmoid_derivative(hiddenLayer[j]) * hiddenLayer[k]
                }
            }

            for(int j=0;j<numHidden;j++){
                for(int k=0;k<numInputs;k++){
                    hiddenWeights[k][j]+=lr*training_inputs[index][k]*hiddenDelta[j]; //Mathematically : hidden weights += (True output - Predicted output) * sigmoid_derivative(outputLayer[j]) * outputWeights[j][k] * sigmoid_derivative(hiddenLayer[j]) * hiddenLayer[k] * training_inputs[index][k]
                }
            }

            for(int j=0;j<numOutputs;j++){
                outputLayerBias[j]+=lr*outputDelta[j];  //Mathematically : outputLayerBias += lr*outputDelta[j]
            }

            for(int j=0;j<numHidden;j++){
                hiddenLayerBias[j]+=lr*hiddenDelta[j]; //Mathematically : hiddenLayerBias += lr*hiddenDelta[j]
            }
        }
    }
    int numHiddenNodes = 1; // Replace 10 with the desired number of hidden nodes
    for (int j = 0; j < numHiddenNodes; j++)
    {
        printf("%f ", hiddenLayerBias[j]);
    }

    fputs("]\nFinal Output Weights", stdout);
    for (int j = 0; j < numOutputs; j++)
    {
        fputs("[ ", stdout);
        for (int k = 0; k < numHiddenNodes; k++)
        {
            printf("%f ", outputWeights[k][j]);
        }
        fputs("]\n", stdout);
    }

    fputs("Final Output Biases\n[ ", stdout);
    for (int j = 0; j < numOutputs; j++)
    {
        printf("%f ", outputLayerBias[j]);
    }

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
            new_hiddenLayer[j] += new_input[k] * hiddenWeights[k][j];
        }
        new_hiddenLayer[j] += hiddenLayerBias[j];
        new_hiddenLayer[j] = sigmoid(new_hiddenLayer[j]);
    }

    for (int j = 0; j < numOutputs; j++)
    {
        new_outputLayer[j] = 0.0;
        for (int k = 0; k < numHidden; k++)
        {
            new_outputLayer[j] += new_hiddenLayer[k] * outputWeights[k][j];
        }
        new_outputLayer[j] += outputLayerBias[j];
        new_outputLayer[j] = sigmoid(new_outputLayer[j]);
    }

    // The result is in new_outputLayer
    double prediction = new_outputLayer[0];

    // Print the prediction
    cout << "Prediction: " << prediction << endl;

    std::ofstream outputFile("weights_and_biases.txt");
    if (outputFile.is_open())
    {
        for (int i = 0; i < numHidden; i++)
        {
            for (int j = 0; j < numInputs; j++)
            {
                outputFile << hiddenWeights[j][i] << " ";
            }
            outputFile << hiddenLayerBias[i] << "\n";
        }
        for (int i = 0; i < numOutputs; i++)
        {
            for (int j = 0; j < numHidden; j++)
            {
                outputFile << outputWeights[j][i] << " ";
            }
            outputFile << outputLayerBias[i] << "\n";
        }
        outputFile.close();
    }
    else
    {
        std::cerr << "Unable to open the output file.\n";
    }
}
