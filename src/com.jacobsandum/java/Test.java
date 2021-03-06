import org.datavec.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.*;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

public class Test {

    private static Logger log = LoggerFactory.getLogger(Test.class);

    static SavedModelBundle bundle;
    static Graph graph;
    static Session session;

    static int height = 28; //Pixel Height
    static int width = 28; //Pixel Width
    static int channels = 1; //Grayscale
    static int outputNum = 10; //10 digits to classify

    public static void main(String[] args) throws IOException, InterruptedException {
        //Fetch exported model and setup variables
        //bundle = SavedModelBundle.load("C:\\Users\\Jacob S\\Documents\\MNIST\\Model\\0.93", "serve");
        bundle = SavedModelBundle.load(System.getProperty("user.dir") + "\\Model", "serve");
        graph = bundle.graph();
        session = bundle.session();
        printModelOperations(graph);

        //Pick an image to test
        INDArray imageToClassify = getNormalizedImage(chooseFile());
        float[][] imageToClassifyFloat = imageToClassify.toFloatMatrix();

        //Expand the matrix by factor of 28, because the model is weird
        //imageToClassifyFloat = expandMatrixHorizontally(imageToClassify);
        //imageToClassifyFloat = expandMatrix(imageToClassifyFloat, 2);
        System.out.println("\nSIZE OF FLOAT MATRIX");
        System.out.println(imageToClassifyFloat.length + "x" + imageToClassifyFloat[0].length);
        //Apply the normalized array to the neural network
        //Create tensor for input from the normalized float matrix
        //Tensor input = Tensors.create(imageToClassifyFloat);
        Tensor input = Tensors.create(imageToClassifyFloat);
        System.out.println("\nINPUT TENSOR");
        System.out.println(input.toString());

        //Create tensor that stores the probabilities result from input tensor
        Tensor result = session.runner().feed("Placeholder:0", input).fetch("Softmax:0").run().get(0);
        System.out.println("\nOUTPUT TENSOR");
        System.out.println(result.toString());

        //Read result tensor
        float[][] resultFloat = new float[1][10];
        result.copyTo(resultFloat);
        System.out.println("\nRESULT FLOAT");
        System.out.println(Arrays.deepToString(resultFloat));

        int[] format = new int[]{0,1,2,3,4,5,6,7,8,9};
        int resultAsInt = 0;
        float greatest = resultFloat[0][0];
        for (int i = 0; i < resultFloat[0].length; i++) {
            if (resultFloat[0][i] > greatest) {
                greatest = resultFloat[0][i];
                resultAsInt = format[i];
            }
        }
        System.out.println("\nFINAL RESULT");
        System.out.println("Digit= " + resultAsInt);
    }

    public static INDArray getNormalizedImage(String path) throws IOException {
        System.out.println("\nNORMALIZING IMAGE AT: " + path);

        //Use NativeImageLoader to convert image file to a matrix that fits the model
        ImageLoader loader = new ImageLoader(height, width, channels);

        //Get the loaded image into an INDArray
        INDArray image = loader.asMatrix(new File(path));

        System.out.println("\nORIGINAL IMAGE AS MATRIX");
        System.out.println(image.toString());

        //Turn values of 0-255 into more values of 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        System.out.println("\nNORMALIZED IMAGE AS MATRIX");
        System.out.println(image.length() + "x");
        System.out.println(image.toString());

        return image;
    }



    public static String chooseFile() {
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION) {
            File file = fc.getSelectedFile();
            String filename = ((File) file).getAbsolutePath();
            return filename;
        } else {
            return null;
        }
    }

    public static void printModelOperations(Graph graph) {
        Iterator itr = graph.operations();
        System.out.println("\nOPERATIONS FOR THIS MODEL");
        while (itr.hasNext()) {
            System.out.println(itr.next());
        }
    }


}
