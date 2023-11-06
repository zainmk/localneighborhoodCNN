/*
 * NetworkBuilder: Using this class to build the network and selectively add the layers we desire in our network. (With the properties we desire)
 */
package network;

import java.util.ArrayList;
import java.util.List;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolLayer;

public class NetworkBuilder {

    private NeuralNetwork net;
    private int inputRows;
    private int inputCols;
    private double scalingFactor;
    List<Layer> layers;

    public NetworkBuilder(int inputRows, int inputCols, double scalingFactor){
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.scalingFactor = scalingFactor;
        this.layers = new ArrayList<>();
    }


    // This will be used to call the constructor for the convolution layer and add it to our network, with the parameters and keeping the current network structure in mind.
    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate){
        if(layers.isEmpty()){
            layers.add(new ConvolutionLayer(filterSize, stepSize, 1, inputRows, inputCols, numFilters, learningRate));
        }
        else{

            //Use the dimensions from the previous layers output to pass through as the input for this new next layer
            Layer prev = layers.get(layers.size() - 1); // This current layer will be the last one in the list as we are building our network.
            layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), numFilters, learningRate));
        }
    }


    // For creating/adding the max pool layer, with similar considerations
    public void addMaxPoolLayer(int stepSize, int windowSize){
        if(layers.isEmpty()){
            layers.add(new MaxPoolLayer(stepSize, windowSize, 1, inputRows, inputCols));
        }
        else{
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));

        }
    }

    // Same for the FullyConnectedLayer (typically placed at the end of the network)
    public void addFullyConnectedLayer(int outputLength, double learningRate){
        if(layers.isEmpty()){
            layers.add(new FullyConnectedLayer(inputCols*inputRows, outputLength, learningRate));
        }
        else{
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new FullyConnectedLayer(prev.getOutputElements(), outputLength, learningRate));
        }
    }

    // TODO: Check if last layer is the fully connected layer, and the output is the right length.

    public NeuralNetwork build(){
        net = new NeuralNetwork(layers, scalingFactor);
        return net;
    }



    















    
}
