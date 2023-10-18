import java.util.List;
import static java.util.Collections.shuffle;

import data.DigitImageReader;
import network.NetworkBuilder;
import data.DigitImage;
import network.NeuralNetwork;

public class Main {
    public static void main (String[] args) {
        
        System.out.println("\nHello World - Welcome to localneighborhoodCNN\n");

        List<DigitImage> digitImages = new DigitImageReader().readData("data/mnist_test.csv");
        
        // Test 1: Testing DigitImage output ~ this should test reading the MNIST data format and printing the digitImages objects created.
        // System.out.println(digitImages.get(0).toString());

        System.out.println("Loading data...");

        List<DigitImage> trainImages = new DigitImageReader().readData("data/mnist_train.csv");
        List<DigitImage> testImages = new DigitImageReader().readData("data/mnist_test.csv");

        // Building the network

        NetworkBuilder builder = new NetworkBuilder(28, 28, 256*100);
        builder.addConvolutionLayer(8, 5, 1, 0.1);
        builder.addMaxPoolLayer(2, 3);
        builder.addFullyConnectedLayer(10, 0.1);

        NeuralNetwork net = builder.build();

        shuffle(trainImages); // shuffle the images so there is no bias in the order they are loaded.
        net.train(trainImages);
        double rate = net.test(testImages);
        System.out.println("Rate after Training: " + rate);


    }
}