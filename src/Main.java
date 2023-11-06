import java.util.List;
import static java.util.Collections.shuffle;

import data.DigitImageReader;
import network.NetworkBuilder;
import data.DigitImage;
import network.NeuralNetwork;

public class Main {
    public static void main (String[] args) {
        
        System.out.println("\nHello World - Welcome to localneighborhoodCNN");
        System.out.println("Loading data...");

        // Load in the 'training' and 'testing' data
        List<DigitImage> trainImages = new DigitImageReader().readData("data/mnist_train.csv");
        List<DigitImage> testImages = new DigitImageReader().readData("data/mnist_test.csv");

        // Building the network
        NetworkBuilder builder = new NetworkBuilder(28, 28, 256*100);
        builder.addConvolutionLayer(8, 5, 1, 0.1);
        builder.addMaxPoolLayer(2, 3);
        builder.addFullyConnectedLayer(10, 0.1);

        NeuralNetwork net = builder.build();

        net.print();

        float rate = net.test(testImages);
        System.out.println("Pre Training Sucess Rate: " + rate);

        shuffle(trainImages);
        net.train(trainImages);

        rate = net.test(testImages);
        System.out.println("Post Training Success Rate: " + rate);

    }
}