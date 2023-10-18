/*
 * Convolution Layer: As the CNN is named after this layer, this layer becomes the core charactersitic of this neural network.
 * The layer consists of a stepping window filter across the inputs and convolutes the output from each, essentially taking a dot product and a measure of how aligned the filter is to the output.
 * The output is high when the filter matches the input and low when the filter does not.
 * This kind of layer is computationaly heavy as it products the filter across the matrices. 
 * 
 * 
 * Backpropagation Alg: The backpropagation for a convolution layer that is trying to optimize differentially with respect to loss.
 * The desire is to find dL/dF (the change of loss with respect to change in filter values) - so we can optimize for a filter value that has minimal loss, thus best suited for the training data.
 * 
 */

package layers;

import java.util.List;
import java.util.Random;

import static utils.MatrixUtility.add;
import static utils.MatrixUtility.flipArrayOnX;
import static utils.MatrixUtility.flipArrayOnY;
import static utils.MatrixUtility.multiply;

import java.util.ArrayList;

public class ConvolutionLayer extends Layer {

    private List<double[][]> filters; // Assuming square filters
    private int filterSize;
    private int stepSize;
    private int inputLength;
    private int inputRows;
    private int inputCols;
    private List<double[][]> prevInput;
    private double learningRate;


    // Consider adding SEED to keep the values consistent 
    public ConvolutionLayer(int filterSize, int stepSize, int inputLength, int inputRows, int inputCols, int numFilters, double learningRate){
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.learningRate = learningRate;

        generateFilters(numFilters);

    }

    public void generateFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();

        Random rand = new Random();

        for(int i=0; i < numFilters; i++){
            double[][] newFilter = new double[filterSize][filterSize]; // square filters
            for(int j=0; j < filterSize; j++){
                for(int k = 0; k < filterSize; k++){

                    newFilter[j][k] = rand.nextGaussian();

                }
            }

            filters.add(newFilter);

        }

        this.filters = filters; // maybe this does not work ?
    }


    public List<double[][]> forwardPass(List<double[][]> list){
        
        List <double[][]> output = new ArrayList<>();
        prevInput = list; // keep a reference for the input (for backpropagation)

        for(int i = 0; i < list.size(); i++){
            for(double[][] filter : this.filters){
                output.add(convolve(list.get(i), filter, stepSize));
            }
        }

        return output;

    }

    public double[][] convolve(double[][] input, double[][] filter, int stepSize){

        int outputRows = (input.length - filter.length)/stepSize + 1;
        int outputCols = (input[0].length - filter[0].length)/stepSize + 1;

        int inputRows = input.length;
        int inputCols = input[0].length;

        int filterRows = filter.length;
        int filterCols = filter[0].length;

        double[][] output = new double[outputRows][outputCols];

        int outputRow = 0;
        int outputCol;

        for(int i = 0; i <= inputRows - filterRows; i+=stepSize){ // Keeps the filter contained within the output matrix.
            
            outputCol = 0;
            
            for(int j = 0; j <= inputCols - filterCols; j+=stepSize){
            
                output[outputRow][outputCol] = applyFilter(input, filter, i, j);
                outputCol++;
            }
            outputRow++;
        }

        return output;
    }

    // Full convolution required when backpropagating the loss to the previous layer.
    public double[][] fullConvolve(double[][] input, double[][] filter){

        int outputRows = (input.length + filter.length) + 1; // Adding the size, as the filter can go over the matrix
        int outputCols = (input[0].length + filter[0].length) + 1;

        int inputRows = input.length;
        int inputCols = input[0].length;

        int filterRows = filter.length;
        int filterCols = filter[0].length;

        double[][] output = new double[outputRows][outputCols];

        int outputRow = 0;
        int outputCol;

        for(int i = -filterRows + 1; i < inputRows; i++){ // Keeps the filter contained within the output matrix. Negative indices dictate a filter value outside of the input matrix.
            outputCol = 0;

            for(int j = -filterCols + 1; j < inputCols; j++){
                output[outputRow][outputCol] = applyFilter(input, filter, i, j);
                outputCol++;

            }
            outputRow++;
        }

        return output;
    }

    public double applyFilter(double[][] input, double[][] filter, int filterPositionX, int filterPositionY){
        double sum = 0.0;
        for(int i = 0; i < filter.length; i++){
            for(int j = 0; j < filter[0].length; j++){
                int inputRowIndex = i + filterPositionX;
                int inputColIndex = j + filterPositionY;

                //Check for when using filters in negative indices (for full convolve, but the filter overlaps the input)
                if(inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inputRows && inputColIndex < inputCols){
                    double value = filter[i][j] * input[inputRowIndex][inputColIndex];
                    sum+= value;
                }
            }
        }
        return sum;
        
    }

    // Spaces the array out and pads with 0's internally, around the step size, this is the resulting shape of the dL/dO, which when convolved with the inputs, will give us the desired dF/dL.
    public double[][] spaceArray(double[][] input){
        if(stepSize == 1){ // No padding required when stepping one by one
            return input;
        }

        int outRows = (input.length - 1)*stepSize + 1;
        int outCols = (input[0].length - 1)*stepSize + 1;

        double[][] output = new double[outRows][outCols];

        // Traversing through the input matrix...
        for(int i = 0; i < input.length; i++){

            for(int j = 0; j < input[0].length; j++){
                output[i*stepSize][j*stepSize] = input[i][j];

            }
        }

        return output;

    }




    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = forwardPass(input);
        return nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, inputLength, inputRows, inputCols);
        return getOutput(matrixInput);
    }


    @Override
    public void backPropagationAlg(double[] dLdO) {
        backPropagationAlg(vectorToMatrix(dLdO, inputLength, inputRows, inputCols));
    }

    @Override
    public void backPropagationAlg(List<double[][]> dLdO) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOprevLayer = new ArrayList<>();

        for(int i = 0; i < filters.size(); i++){
            filtersDelta.add(new double[filterSize][filterSize]);
        }

        for(int i = 0; i < prevInput.size(); i++){

            double[][] errorForInput = new double[inputRows][inputCols];

            for(int j = 0; j < filters.size(); j++){

                double[][] currentFilter = filters.get(j);
                double[][] error = dLdO.get(i*filters.size() + j);

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(prevInput.get(i), spacedError, 1); // convolve with step 1, as array is spaced out already

                double[][] delta = multiply(dLdF, learningRate*-1); // Apply a negative learning rate to shrink the filter depending on loss.
                double[][] newTotalDelta = add(filtersDelta.get(j), delta);
                filtersDelta.set(j, newTotalDelta);

                // Flip the spaced error in the format we desire, then fully convolve with the current filter we're looking at, then add all the input errors up for that filter.
                errorForInput = add(errorForInput, fullConvolve(currentFilter, flipArrayOnX(flipArrayOnY(spacedError))));

            }

            dLdOprevLayer.add(errorForInput);

        }

        // Update the filters with the new total delta loss (each filter cumulates the deltas).
        for(int i = 0; i < filters.size(); i++){
            double[][] adjusted = add(filtersDelta.get(i), filters.get(i)); // Although adding, recall the *-1 factor, therefore applying 'loss' to filter
            filters.set(i, adjusted);
        }

        if(previousLayer != null){
            previousLayer.backPropagationAlg(dLdOprevLayer);
        }

    }

    @Override
    public int getOutputLength() {
        return inputLength*filters.size();
    }

    @Override
    public int getOutputRows() {
        return (inputRows - filterSize)/stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inputCols - filterSize)/stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputLength()*getOutputRows()*getOutputCols();
    }
    
}
