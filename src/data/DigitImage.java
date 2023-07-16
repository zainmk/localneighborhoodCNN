/*
 * Function: Represent a digit image object contained by the MNIST data set 
 * digitData: A 2D Array holding the actual data that represents the visial image (the pixel value)
 * digitLabel: The identified digit integer for that digit data [0-9].
 * 
*/

package data;

public class DigitImage {

    private double[][] digitData;
    private int digitLabel;

    public double[][] getDigitData(){
        return digitData;
    }

    public int getDigitLabel(){
        return digitLabel;
    }
    
    public DigitImage(double[][] data, int label) {
        this.digitData = data;
        this.digitLabel = label;
    }
    
    @Override // we override the toString method to print accordingly on digitImage objects.
    public String toString(){
        String s = digitLabel + ", \n";
        for(int i = 0; i < digitData.length; i++){
            for(int j = 0; j < digitData[0].length; j++){
                s += digitData[i][j] + ", ";   
            }
            s += "\n";
        }
        return s;
    }

}
