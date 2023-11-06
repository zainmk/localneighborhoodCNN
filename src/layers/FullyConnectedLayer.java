/* 
 * FullyConnectedLayer (Layer)
 * One of the layers implemented will be a fully connected layer that will have all outputs connected to the inputs of the next layer.
 * 
 * Each node of this input layer will have weights associated with it, corresponding to the overall cost-effective function. 
 * 
 * We will also be using RELU activation functions, to calibrate the weights of the network and align the correction to be only in the positive.
 */

package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {

    private int inputLength;
    private int outputLength;

    private double[] lastOutput;
    private double[] lastInput;
    
    private double learningRate;

    private final double LEAK = 0.01; // This will help with nudging the RELU derviative, so it doesn't just give a 0. Any small non-zero value
    
    private double[][] weights; // These will hold the weights of the nodes of this connected layer. There is a weight for every 'edge' connecting the nodes of the layer.

    // Constructor - when defined, needs to be defined with the input and output lengths given. These are charactersitic and required for a 'FullyConnectedLayer'
    public FullyConnectedLayer(int inputLength, int outputLength, double learningRate){
        this.inputLength = inputLength; // Define the number of input nodes for the layer.
        this.outputLength = outputLength; // Define the number of output nodes for the layer.
        this.weights = new double[inputLength][outputLength]; // Create the size of the weights matrix depending on the # of inputs/outputs.
        this.learningRate = learningRate;

        randomizeWeights();
    }

    // Purpose: For the fully connected layer, the getOutput will make use of our implemented 'forwardpass' but can still take the matrix of weights as input.
    //          For invoking with a double[][], we linearize the matrix to a single vector and pass that along to associated 'getOutput' method.
    @Override
    public double[] getOutput(List<double[][]> input) {
        return getOutput(matrixToVector(input));
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = forwardPass(input); // Create the forward pass of the network on the first go.

        // If there is a next layer, the output must get processed by the next network layer so we call the 'getOutput' on the next one
        // otherwise, the output is the final output and the forwardpass is just returned.
        
        return (nextLayer != null) ? nextLayer.getOutput(forwardPass) : forwardPass;
    }

    // Purpose: Implementing the backpropagation for the fully connected layer
    // Backpropagating requires deriving the loss function with respect to the weights (so we can optimize these weights)...
    // dL/dW = dL/dO * dO/dZ * dZ/dW (recall Z being the intermediate output - before loss function) where we desire the dLdO
    @Override
    public void backPropagationAlg(double[] dLdO) {

        double[] dLdX = new double[inputLength]; 

        // Derivative of the final output with respect to the intermediate output (before loss function)
        double dOdZ;

        // Derivative of the intermediate output with respect to the weights used to get there
        double dZdW;

        // Derivative of the final loss with respect to the weights of the matrix (the intended derivative)
        double dLdW;

        // Derivative of the final output with respect to the input (the intended final output)
        double dZdX;

        for(int i = 0; i < inputLength; i++){

            double dLdX_sum = 0;

            for(int j = 0; j < outputLength; j++){

                dOdZ = d_relu(lastOutput[j]);
                dZdW = lastInput[i];
                dZdX = weights[i][j];

                dLdW = dLdO[j]*dOdZ*dZdW;

                // We use the learning rate here to allow it configure to a particular value but still be dynamic enough to be trainable for all sets of data.
                weights[i][j] -= dLdW*learningRate; 

                //We calculate the sum of the derivative of loss to input
                dLdX_sum += dLdO[j]*dOdZ*dZdX;

            }

            dLdX[i] = dLdX_sum;

        }

        if(previousLayer != null){
            previousLayer.backPropagationAlg(dLdX);
        }
    }

    @Override
    public void backPropagationAlg(List<double[][]> dLdO) {
        backPropagationAlg(matrixToVector(dLdO));
    }

    // The intended output for this layer is a one dimensional vector, therefore the rows, columns and length are all 0 while the elements are N.

    @Override
    public int getOutputLength() {
       return 0;
    }

    @Override
    public int getOutputRows() {
       return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return outputLength;
    }

    @Override
    public void print(){
        System.out.println("Fully Connected Layer: \nInputLength:" + inputLength + "\nOutputLength: " + outputLength);
        System.out.println("Weights...");
        
        for(double[] weight: weights){
            for(double w: weight){
                System.out.print(w + " , ");
            }
            System.out.println();
        }
     
    }

    // The initial weights of the network need to be randomized for an 'unbiased' starting point where the network can 'start' to train from.
    // We use the Gaussian randomg values so the weights are well distributed but within +/- 1 (following the Gaussian distribution).
    // TEST - Try different initial weight setups to see how they affect the networks output and speed. Note that a fixed SEED would generate the same random values.
    public void randomizeWeights(){
        Random rand = new Random();
        for(int i = 0; i < inputLength; i++){
            for(int j = 0; j < outputLength; j++){
                weights[i][j] = rand.nextGaussian();
            }
        }
    }

    // Purpose: The forward pass is the method used to translate the output values through the weights of the network, from initial input to final output.
    //          The overall equation is therefore a summation of the input values with the product of their corresponding weights.
    //          The output afterwards is then put through the activation function filter.
    // Inputs: A 'double[]' of values corresponding to the 'developing' classification of the network.
    // Outputs: The 'double[]' of values (similar format to input) used as either the final output to be applied to the activation function, or fed into more networks.
    public double[] forwardPass(double[] input){

        lastInput = input;

        double[] z_output = new double[outputLength];
        double[] relu_output = new double[outputLength]; // Applying activation function (relu) to the output nodes.

        for(int i = 0; i < inputLength; i++){
            for(int j = 0; j < outputLength; j++){
                z_output[j] += input[i]*weights[i][j]; // size of output is # of output nodes (j)
            }
        }

        lastOutput = z_output; // Need a seperate reference to keep a hold of the unfilteredOutput.

        for(int i = 0; i < inputLength; i++){
            for(int j = 0; j < outputLength; j++){
                relu_output[j] = relu(z_output[j]);
            }
        }

        return relu_output;
    }

    // Purpose: RELU Activation Function: Help filter the weights and make finding the optimal ones easier. Not necessary, but helps to escape 'deadzones' during weight optimization.
    //          The function will introduce some non-linearities in the resulting equation which will make derivations during backpropagation cleaner.
    //          Note the following simplicity of the derviation for RELU as well.
    // Input: The raw output value of the network after the inputs have been propagated through the forward pass.
    // Output: The RELU filtered value of the network, to be used as the final output of the network or before passing along to the next layer.
    // TEST: Different activation functions and how they effect the network's speed and accuracy.
    public double relu(double input){
        if(input <= 0){
            return 0;
        }
        else{
            return input;
        }
    }

    public double d_relu(double input){ // Derivative of RELU
        if(input <= 0){
            return LEAK; // Leak used to eliminate dead zones as the weights correct based off the loss
        }
        else{
            return 1;
        }
    }
    
}
