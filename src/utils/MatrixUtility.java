/*
 * MatrixUtility ~ A set of static utility methods to use with regards to the necessary matrix math required.
 *  
 */


package utils;

public class MatrixUtility {

    // Adding Matrices
    public static double[][] add(double[][] a, double[][] b){
        double[][] output = new double[a.length][a[0].length];
        for(int i = 0; i < a.length; i++){
            for(int j = 0; j < a[0].length; j++){

                output[i][j] = a[i][j] + b[i][j];

            }
        }
        return output;
    }


    // Adding Vectors
    public static double[] add(double[] a, double[] b){
        double[] output = new double[a.length];
        for(int i = 0; i < a.length; i++){
            output[i] = a[i] + b[i];
        }
        return output;
    }

    //Multiplying Matrix by Scalar
    public static double[][] multiply(double[][] a, double scalar){
        double[][] output = new double[a.length][a[0].length];
        for(int i = 0; i < a.length; i++){
            for(int j = 0; j < a[0].length; j++){

                output[i][j] = a[i][j]*scalar;

            }
        }
        return output;

    }


    //Multiplying Vector by Scalar
    public static double[] multiply(double[] a, double scalar){
        double[] output = new double[a.length];
        for(int i = 0; i < a.length; i++){
            output[i] = a[i]*scalar;
        }

        return output;
    }

    public static double[][] flipArrayOnX(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                output[rows - i - 1][j] = array[i][j];
            }
        }

        return output;

    }

    public static double[][] flipArrayOnY(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){

                output[i][cols - j - 1] = array[i][j];

            }
        }

        return output;

    }


    // Helper function to find the maximum index as per its value.
    public static int getMaxIndex(double[] input){
        double max = 0;
        int index = 0;

        for(int i = 0; i < input.length; i++){
            if(input[i] >= max){
                max = input[i];
                index = i;
            }
        }

        return index;
    }




}
