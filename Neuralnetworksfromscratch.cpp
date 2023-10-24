#include <iostream>
#include <math.h> 
#include <vector>
#include <random>

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
double init_weights(){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}