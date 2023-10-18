/* NeuralNetwork: Will consist of the differnet kinds of layers that make up the neural network and are linked together, from input of digit images to output of a classification.
 * This will give us the guess of the network and how well it is trained. 
 */

package network;


import static utils.MatrixUtility.add;
import static utils.MatrixUtility.multiply;
import static utils.MatrixUtility.getMaxIndex;

import layers.Layer;
import data.DigitImage;
import java.util.List;
import java.util.ArrayList;

public class NeuralNetwork {

    List<Layer> layers;
    double scalingFactor; // TODO: Consider building/using a 'scaling' layer instead (helps reduce the input size and stabilize the network changes)

    public NeuralNetwork(List<Layer> layers, double scalingFactor){
        this.layers = layers;
        this.scalingFactor = scalingFactor;
        linkLayers();

    }

    // This will call on the getter/setters for the layer to link them for the neural network
    private void linkLayers(){

        if(layers.size() == 1){
            return ;
        }

        for(int i = 0; i < layers.size(); i++){
            if(i == 0){
                layers.get(i).set_nextLayer(layers.get(i + 1));
            }
            else if (i == layers.size() - 1){
                layers.get(i).set_previousLayer(layers.get(i - 1));
            }
            else{
                layers.get(i).set_previousLayer(layers.get(i - 1));
                layers.get(i).set_nextLayer(layers.get(i  + 1));
            }

        }

    }

    
    // This will give us the error of the neural network from which we can evaluate how well the network has been trained on the training data.
    public double[] getError(double[] networkOutput, int correctAnswer){

        int numOfClasses = networkOutput.length; // This should correspond to the number of possible classifications there can be (in this case, 0-9 (10))

        double[] expected = new double[numOfClasses];
        expected[correctAnswer] = 1;

        // To find the actual error, we find the difference between the output and the expected value.
        return add(networkOutput, multiply(expected, -1));
    }


    public int guess(DigitImage image){
        
        List<double[][]> inputList = new ArrayList<>();

        // Converting the image into a list format to use with the neural network.
        inputList.add(multiply(image.getDigitData(), (1.0/scalingFactor)));

        // Calling it on the first layer, will allow it to propagate through the layers.
        double[] output = layers.get(0).getOutput(inputList);

        return getMaxIndex(output); // The max index is the actual digit number value.
    }


    public float test(List<DigitImage> images){
        int correct = 0;

        for(DigitImage image: images){
            int guess = guess(image);

            if(guess == image.getDigitLabel()){
                correct++;
            }
        }

        return (float)(correct/images.size()); // Perecentage of correct.
    }

    public void train(List<DigitImage> images){

        for(DigitImage image:images){
            List<double[][]> inputList = new ArrayList<>();
            inputList.add(multiply(image.getDigitData(), (1.0/scalingFactor)));

            double[] output = layers.get(0).getOutput(inputList); // Again, the output will propagate through the connected layers.
            double[] dLdO = getError(output, image.getDigitLabel());

            layers.get((layers.size() - 1)).backPropagationAlg(dLdO); // Update the weights based on the calculated errors, with the relevant backPropagation (starting from the last layer ~ which will also 'propagate backwards' through the layers)

        }

    }

    // Print function for the network to show the network and aid in debugging.
    public void print(){
        for(Layer layer: layers){
            System.out.println(layer);
        }
    }




}

