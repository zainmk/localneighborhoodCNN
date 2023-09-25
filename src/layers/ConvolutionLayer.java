/*
 * Convolution Layer: As the CNN is named after this layer, this layer becomes the core charactersitic of this neural network.
 * The layer consists of a stepping window filter across the inputs and convolutes the output from each, essentially taking a dot product and a measure of how aligned the filter is to the output.
 * The output is high when the filter matches the input and low when the filter does not.
 * This kind of layer is computationaly heavy as it products the filter across the matrices. 
 * 
 */


package layers;

import java.util.List;
import java.util.Random;
import java.util.ArrayList;

public class ConvolutionLayer extends Layer {

    private List<double[][]> filters; // Assuming square filters
    private int filterSize;
    private int stepSize;
    private int inputLength;
    private int inputRows;
    private int inputCols;


    // Consider adding SEED to keep the values consistent 
    public ConvolutionLayer(int filterSize, int stepSize, int inputLength, int inputRows, int inputCols, int numFilters){
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;

        generateFilters(numFilters);

    }

    public void generateFilters(int numFilters){
        List<double[][]> filters = new ArrayList();

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
            for(int j = 0; j < inputCols - filterCols; j+=stepSize){
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

                double value = filter[i][j] * input[inputRowIndex][inputColIndex];
                sum+= value;
            }
        }
        return sum;
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
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagationAlg'");
    }

    @Override
    public void backPropagationAlg(List<double[][]> dLdO) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagationAlg'");
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
