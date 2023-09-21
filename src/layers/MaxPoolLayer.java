/*
 * Max Pool Layer: The max pool layer uses a lesser dimensional filter to summarize the features and assign the weights.
 * - They reduce the dimensions of the feature map therefore the number of parameters to optimize for.
 * - They help average out the analysis done by a convolutional layer before it is processed again through another. This allows for less specifically trained data (overfitting).
 * - Information is loss by pooling results like this, fine grained details can be lost in terms of minor features and tuning the size/stride can take more time.
 * - When it comes to derivations, we use the chain rule on the max sum of each partition of the feature map.
 */

package layers;

import java.util.List; // Fixed Positonal Reference
import java.util.ArrayList; // Dynamic Sizing

public class MaxPoolLayer extends Layer {

    private int stepSize;
    private int windowSize;
    private int inputLength;
    private int inputRows;
    private int inputCols;

    // Track a record of the positions of the max values, to derive for loss in backpropagation.
    List <int[][]> lastMaxRows;
    List <int[][]> lastMaxCols;

    public MaxPoolLayer(int stepSize, int windowSize, int inputLength, int inputRows, int inputCols){
        this.stepSize = stepSize;
        this.windowSize = windowSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;

    }

    public List<double[][]> forwardPass(List<double[][]> input){

        List<double[][]> output = new ArrayList<>();
        lastMaxRows = new ArrayList<>();
        lastMaxCols = new ArrayList<>();

        for(int l=0; l < input.size(); l++){
            output.add(pool(input.get(l)));
        }

        return output;
    }

    public double[][] pool(double[][] input){

        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];


        for(int i = 0; i < getOutputRows(); i += stepSize){
            for(int j = 0; j < getOutputCols(); j += stepSize){

                //Now within the pooling window...

                double max = 0.0;
                maxRows[i][j] = -1;
                maxCols[i][j] = -1;

                for(int k = 0; k < windowSize; k++){
                    for(int l = 0; l < windowSize; l++){

                        if(max < input[i + k][j + l]){
                            max = input[i + k][j + l];
                            maxRows[i][j] = i + k;
                            maxCols[i][j] = j + l;
                        }


                    }
                }

                output[i][j] = max;

            }
        }

        lastMaxRows.add(maxRows);
        lastMaxCols.add(maxCols);

        return output;
        
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = forwardPass(input);
        return (nextLayer != null) ? nextLayer.getOutput(outputPool) : getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, inputLength, inputRows, inputCols);
        return getOutput(matrixList);

    }

    @Override
    public void backPropagationAlg(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagationAlg(matrixList);
    }

    @Override
    public void backPropagationAlg(List<double[][]> dLdO) {
        List<double[][]> dXdL = new ArrayList<>();
        int l = 0;

        for(double[][] LossOutputArray : dLdO){
            double[][] error = new double[inputRows][inputCols];
            for(int i = 0; i < getOutputRows(); i++){
                for(int j = 0; j < getOutputCols(); j++){

                    int maxRow = lastMaxRows.get(l)[i][j];
                    int maxCol = lastMaxCols.get(l)[i][j];

                    if(maxRow != -1){
                        error[maxRow][maxCol] += LossOutputArray[i][j]; // Add the losses when selected more than once in the pool.
                    }

                }
            }

            dXdL.add(error); // Add the error from the input with respect to Loss
            l++;

        }

        if(previousLayer != null){
            previousLayer.backPropagationAlg(dXdL);
        }

    }

    @Override
    public int getOutputLength() {
        return inputLength;
    }

    @Override
    public int getOutputRows() {
        return (inputRows - windowSize)/stepSize + 1; // How to obtain the number of resulting rows after applying the filter (based on window and step size)
    }

    @Override
    public int getOutputCols() {
        return (inputCols - windowSize)/stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return inputLength*getOutputRows()*getOutputCols();
    }
    
}
