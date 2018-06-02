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
        //bundle = SavedModelBundle.load("C:\\Users\\Jacob S\\Documents\\MNIST\\Model\\1527827482", "serve");
        bundle = SavedModelBundle.load("C:\\Users\\Jacob S\\Documents\\MNIST\\Model\\1527903533", "serve");
        //bundle = SavedModelBundle.load("C:\\Users\\Jacob S\\Desktop\\karl\\model", "serve");
        graph = bundle.graph();
        session = bundle.session();
        printModelOperations(graph);
        //Pick an image to test
        INDArray imageToClassify = getNormalizedImage(chooseFile());
        float[][] imageToClassifyFloat = imageToClassify.toFloatMatrix();
        System.out.println("SIZE OF FLOAT MATRIX");
        System.out.println(imageToClassifyFloat.length + "x" + imageToClassifyFloat[0].length);
        //Apply the normalized array to the neural network
        //Create tensor for input from the normalized float matrix
        Tensor input = Tensors.create(imageToClassifyFloat);
        System.out.println("\nINPUT TENSOR");
        System.out.println(input.toString());
        //Create tensor that stores the result from input tensor
        Tensor result = session.runner().feed("Placeholder:0", input).fetch("Softmax:0").run().get(1);
        System.out.println("\nOUTPUT TENSOR");
        System.out.println(result.toString());

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

    public static void printModelOperations(SavedModelBundle savedModelBundle) {
        Iterator itr = savedModelBundle.graph().operations();
        System.out.println("\nOPERATIONS FOR THIS MODEL");
        while (itr.hasNext()) {
            System.out.println(itr.next());
        }
    }

}
