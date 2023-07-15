/*
 * The DigitImageReader will be used to read the digitImage objects. It will hold a list of these image objects.
 * 
 * To read the image data, each digitImage is conatined within each line of the MNIST data set. The first digit of the line denotes the 'label'
 * while the remaining (784) digits on the line denote the pixel value.
 */

package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;


//trying a new way to read the file
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;


public class DigitImageReader {

    // They will always have a fixed size, as the pixel images are of a fixed size for each digit image (28 x 28).
    private final int rows = 28;
    private final int cols = 28;

    public List<DigitImage> readData(String path){ // This method will be used to read the actual data of MNIST data set


        List<DigitImage> digitImages = new ArrayList<>();
        File inputDataFile = new File(path);

        try{
            Scanner myReader = new Scanner(inputDataFile);
            
            while (myReader.hasNextLine()) {
                String[] lineItems = myReader.nextLine().split(",");
                // String[] lineItems = line.split(","); // As the data points are 'comma separated', we split the input line by the commas
                double data[][] = new double[rows][cols]; // To be used as the 'digitData' that represents our DigitImage
                int label = Integer.parseInt(lineItems[0]); // Grab the first value of the line, which should be the label; to be used as the 'digitLabel' that represents our digitImage
                int index = 1; // represents the pixel value we are indexing within the line of values

                for(int i = 0; i < rows; i++){
                    for(int j = 0; j < cols; j++){
                        data[i][j] = (double) Integer.parseInt(lineItems[index]);
                        index++;
                    }
                }
                digitImages.add(new DigitImage(data, label)); // Add the new digit image in the object representation.
            }

            myReader.close();

        }
        catch(FileNotFoundException e){
            System.out.println("Error - File not found");
        }


        // This will read from the file path of where we keep the MNIST data
        // try (BufferedReader dataReader = new BufferedReader(new FileReader(path)) ){

        //     String line;

        //     while((line = dataReader.readLine()) != null){ // go through all of the lines the dataReader has picked up (each line is a new digitImage pixel entry)
        //         String[] lineItems = line.split(","); // As the data points are 'comma separated', we split the input line by the commas
        //         double data[][] = new double[rows][cols]; // To be used as the 'digitData' that represents our DigitImage
        //         int label = Integer.parseInt(lineItems[0]); // Grab the first value of the line, which should be the label; to be used as the 'digitLabel' that represents our digitImage
        //         int index = 1; // represents the pixel value we are indexing within the line of values

        //         for(int i = 0; i < rows; i++){
        //             for(int j = 0; j < cols; j++){
        //                 data[i][j] = (double) Integer.parseInt(lineItems[index]);
        //                 index++;
        //             }
        //         }
        //         digitImages.add(new DigitImage(data, label)); // Add the new digit image in the object representation.

        //     }

        // } catch (Exception e){
        //     throw new IllegalArgumentException("File not found " + path);
        // }

        return digitImages;
    }
}
