import java.util.List;

import data.DigitImageReader;
import data.DigitImage;

public class Main {
    public static void main (String[] args) {
        
        
        System.out.println("Hello World - Welcome to localneighborhoodCNN\n");

        List<DigitImage> digitImages = new DigitImageReader().readData("data/mnist_test.csv");
        
        // Test 1: Testing DigitImage output ~ this should test reading the MNIST data format and printing the digitImages objects created.
        System.out.println(digitImages.get(0).toString());


    }
}