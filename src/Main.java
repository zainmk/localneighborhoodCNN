import java.util.List;

import data.DigitImageReader;
import data.DigitImage;

public class Main {
    public static void main (String[] args) {
        
        
        System.out.println("Hello World - Welcome to localneighborhoodCNN\n");

        List<DigitImage> digitImages = new DigitImageReader().readData("data/mnist_test.csv");
        System.out.printf(digitImages.get(0).toString());


    }
}