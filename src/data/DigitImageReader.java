/*
 * The DigitImageReader will be used to read the digitImage objects. It will hold a list of these image objects.
 * 
 * To read the image data, each digitImage is conatined within each line of the MNIST data set. The first digit of the line denotes the 'label'
 * while the remaining (784) digits on the line denote the pixel value.
 */

package data;

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class DigitImageReader {

    // They will always have a fixed size, as the pixel images are of a fixed size for each digit image (28 x 28).
    private final int rows = 28;
    private final int cols = 28;

    public List<DigitImage> readData(String path){
        
        List<DigitImage> digitImages = new ArrayList<>();
        File inputDataFile = new File(path);

        try{
            Scanner fileReader = new Scanner(inputDataFile);
            while (fileReader.hasNextLine()) {
                String[] lineItems = fileReader.nextLine().split(",");
                double data[][] = new double[rows][cols]; // To be used as the 'digitData' that represents our DigitImage
                int label = Integer.parseInt(lineItems[0]); // Grab the first value of the line, which should be the label; to be used as the 'digitLabel' that represents our digitImage
                int index = 1; // will index the pixel value we are indexing within the line of values

                for(int i = 0; i < rows; i++){
                    for(int j = 0; j < cols; j++){
                        data[i][j] = (double) Integer.parseInt(lineItems[index]);
                        index++;
                    }
                }
                digitImages.add(new DigitImage(data, label)); // Add the new digit image in the object representation.
            }

            fileReader.close();

        }
        catch(FileNotFoundException e){
            System.out.println("Error - File not found");
            e.printStackTrace();
        }

        return digitImages;
    }
}
