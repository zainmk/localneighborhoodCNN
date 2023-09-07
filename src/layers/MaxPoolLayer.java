/*
 * Max Pool Layer: The max pool layer uses a lesser dimensional filter to summarize the features and assign the weights.
 * - They reduce the dimensions of the feature map therefore the number of parameters to optimize for.
 * - They help average out the analysis done by a convolutional layer before it is processed again through another. This allows for less specifically trained data.
 * - Information is loss by pooling results like this, fine grained details can be lost in terms of minor features and tuning the size/stride can take more time.
 * - When it comes to deriving the values, we use the chain rule on the max sum of each partition of the feature map.
 */

package layers;

import java.util.List;

public class MaxPoolLayer extends Layer {

    @Override
    public double[] getOutput(List<double[][]> input) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
    }

    @Override
    public double[] getOutput(double[] input) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
    }

    @Override
    public void backPropagationAlg(double[] dldO) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagationAlg'");
    }

    @Override
    public void backPropagationAlg(List<double[][]> dldO) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagationAlg'");
    }

    @Override
    public int getOutputLength() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutputLength'");
    }

    @Override
    public int getOutputRows() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutputRows'");
    }

    @Override
    public int getOutputCols() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutputCols'");
    }

    @Override
    public int getOutputElements() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutputElements'");
    }
    
}
