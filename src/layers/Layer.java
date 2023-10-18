/*
 * Layer (Abstract) ~ This will be the general form of our layers, of which the computations will occur from within.
 * Each layer will have inputs that can come from another layer or outputs to give to another layer.
 */

package layers;

import java.util.List;
import java.util.ArrayList;

public abstract class Layer {

    // Getters and Setters for our protected attributes for the abstract Layer.
    public Layer get_nextLayer(){
        return nextLayer;
    }
    public Layer get_previousLayer(){
        return previousLayer;
    }
    public void set_nextLayer(Layer nextLayer){
        this.nextLayer = nextLayer;
    }
    public void set_previousLayer(Layer previousLayer){
        this.previousLayer = previousLayer;
    }

    // Will need a reference to the layer in front and behind. This will allow us to follow the chain
    // of layers from the beginning to the end and follow the processing flow.
    protected Layer nextLayer;
    protected Layer previousLayer;

    // Defining some abstract classes that will be required of all layer types...

    // Input: Convolutional layer nodes take a list of matrices (pixel image data) while a simple connected layer would have 
    //        just a vector of the original classification attempt being made (0-9).
    // Output: Networks best attempt at classification of the digit - confidence of the classification of each digit. (double[])
    //         Even the networks knows how far off it's estimate is.
    // Purpose: Need to convert the input data types appropriately based on what kind of input the layer is dealing with and produce the output 'double[]' to give to
    //          the next layer for processing or as the final output
    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    // Back Propagation is the underlying algorithm in the neural network that optimizes the models parameters and produces a network well trained on the data.
    // Input: As this applies to our layers, it must act on the data types these layers support such as the weights of the classification (double[]) or the image data (List<double[][]>)
    //        The input itself represents the ratio of change of loss (cost) with respect to the resulting output of the model. These ratios help adjust the weights accordingly
    //        and allow the network model's parameters to efficiently reach the optimal values.
    // Output: There is no output for the backpropagation as it will update respective layer's weights (and all following as connected).
    // Purpose: Goes back up the stack of layers being used and updates the weights accordingly based off the errors
    public abstract void backPropagationAlg(double[] dLdO);
    public abstract void backPropagationAlg(List<double[][]> dLdO);


    // Input: No inputs for these methods as they simlpy output charactersitics of the layer.
    // Output: We need to quickly obtain the list of matrices length, the number of rows/columns, and the number of elements in total from the layer. Implemented differently for each layer.
    // Purpose: We require methods to extract the outputs from the previous layers. Given the size of the layers, we want some charactersitics of the size to keep track of.
    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElements();

    // Input: The matrix of image data List<double[][]>
    // Output: The corresponding one dimensional vector that has all the image data values stretched out (double[])
    // Purpose: We need to be able to convert the list of double[][] that represent the image data of pixels into a one dimensional vector that is compatible with our layer type and our intended ouput of classification.
    //          This will be used by all the layers and therefore is implemented here (and not kept as abstract)
    
    
    // TODO: Move the following methods to the matrix utility.

    public double[] matrixToVector(List<double[][]> input){
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;
        
        double[] vector = new double[length*rows*cols];
        // breaking out the List<double[][]> into a double[][] and further into the double[], then storing that double.
        int i = 0;
        for(double[][] ListItem : input){
            for(double[] arrayItem : ListItem){
                for(double item : arrayItem){
                    vector[i] = item;
                    i++;

                }
            }
        }

        return vector;
    }


    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols){
        List<double[][]> output = new ArrayList<>();

        if(input.length != rows*cols*length){ // bound checking to see we have enough values
            throw new IllegalArgumentException("Invalid dimensions provided");
        }

        int i = 0;
        for(int l = 0; l < length; l++){
            double[][] matrix = new double[rows][cols];
            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    matrix[r][c] = input[i];
                    i++;
                }
            }
            output.add(matrix); // Add the resulting double[][] to the output list.
        }
        return output;
    }






    
}
