package com.sudheerherle.ObjectRecognition;

import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.TabPane;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 *
 * @author Sudheer Herle 
 * Website: http://www.sudheerherle.com 
 * Email : hello@sudheerherle.com
 * Created on: July 10, 2018
 * Download the pre-trained inception model from here: https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip 
 */
public class Recognizer extends Application {


    private JButton predict;
    private JButton incep;
    private JButton img;
    private JFileChooser incepch;
    private JFileChooser imgch;
    private JLabel viewer;
    private JTextField result;
    private JTextField imgpth;
    private JTextField modelpth;
    private FileNameExtensionFilter imgfilter = new FileNameExtensionFilter(
            "JPG & JPEG Images", "jpg", "jpeg");
    private String modelpath;
    private String imagepath;
    private boolean modelselected = false;
    private byte[] graphDef;
    private List<String> labels;

    public Recognizer() {
        
        predict = new JButton("Predict");
        predict.setEnabled(false);
        incep = new JButton("Choose Inception");
        img = new JButton("Choose Image");
        
        incepch = new JFileChooser();
        imgch = new JFileChooser();
        imgch.setFileFilter(imgfilter);
        imgch.setFileSelectionMode(JFileChooser.FILES_ONLY);
        incepch.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        
        result=new JTextField();
        modelpth=new JTextField();
        imgpth=new JTextField();
        modelpth.setEditable(false);
        imgpth.setEditable(false);
        viewer = new JLabel();
    }

//    public void actionPerformed(ActionEvent e) {
//
//        if (e.getSource() == incep) {
//            int returnVal = incepch.showOpenDialog(this);
//
//            if (returnVal == JFileChooser.APPROVE_OPTION) {
//                File file = incepch.getSelectedFile();
//                modelpath = file.getAbsolutePath();
//                modelpth.setText(modelpath);
//                System.out.println("Opening: " + file.getAbsolutePath());
//                modelselected = true;
//                graphDef = readAllBytesOrExit(Paths.get(modelpath, "tensorflow_inception_graph.pb"));
//                labels = readAllLinesOrExit(Paths.get(modelpath, "imagenet_comp_graph_label_strings.txt"));
//            } else {
//                System.out.println("Process was cancelled by user.");
//            }
//
//        } else if (e.getSource() == img) {
//            int returnVal = imgch.showOpenDialog(Recognizer.this);
//            if (returnVal == JFileChooser.APPROVE_OPTION) {
//                try {
//                    File file = imgch.getSelectedFile();
//                    imagepath = file.getAbsolutePath();
//                    imgpth.setText(imagepath);
//                    System.out.println("Image Path: " + imagepath);
//                    Image img = ImageIO.read(file);
//
//                    viewer.setIcon(new ImageIcon(img.getScaledInstance(200, 200, 200)));
//                    if (modelselected) {
//                        predict.setEnabled(true);
//                    }
//                } catch (IOException ex) {
//                    Logger.getLogger(Recognizer.class.getName()).log(Level.SEVERE, null, ex);
//                }
//            } else {
//                System.out.println("Process was cancelled by user.");
//            }
//        } else if (e.getSource() == predict) {
//            byte[] imageBytes = readAllBytesOrExit(Paths.get(imagepath));
//
//            try (Tensor image = Tensor.create(imageBytes)) {
//                float[] labelProbabilities = executeInceptionGraph(graphDef, image);
//                int bestLabelIdx = maxIndex(labelProbabilities);
//                result.setText("");
//                result.setText(String.format(
//                                "BEST MATCH: %s (%.2f%% likely)",
//                                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
//                System.out.println(
//                        String.format(
//                                "BEST MATCH: %s (%.2f%% likely)",
//                                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
//            }
//
//        }
//    }

   
    
    @Override
    public void start(Stage primaryStage) throws Exception {
        try {			
			FXMLLoader loader = new FXMLLoader(Recognizer.class.getResource("/fxml/ObjectRecognizer.fxml"));
			AnchorPane rootElement = loader.load();
			Scene scene = new Scene(rootElement);			
			primaryStage.setTitle("Object Recognition");
			primaryStage.setScene(scene);
			primaryStage.show();	
                        primaryStage.setResizable(false);
			
		} catch(Exception e) {
			e.printStackTrace();
		}
    }

    // In the fullness of time, equivalents of the methods of this class should be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
    // like Python, C++ and Go.
    static class GraphBuilder {

        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output div(Output x, Output y) {
            return binaryOp("Div", x, y);
        }

        Output sub(Output x, Output y) {
            return binaryOp("Sub", x, y);
        }

        Output resizeBilinear(Output images, Output size) {
            return binaryOp("ResizeBilinear", images, size);
        }

        Output expandDims(Output input, Output dim) {
            return binaryOp("ExpandDims", input, dim);
        }

        Output cast(Output value, DataType dtype) {
            return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
        }

        Output decodeJpeg(Output contents, long channels) {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                    .addInput(contents)
                    .setAttr("channels", channels)
                    .build()
                    .output(0);
        }

        Output constant(String name, Object value) {
            try (Tensor t = Tensor.create(value)) {
                return g.opBuilder("Const", name)
                        .setAttr("dtype", t.dataType())
                        .setAttr("value", t)
                        .build()
                        .output(0);
            }
        }

        private Output binaryOp(String type, Output in1, Output in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
        }

        private Graph g;
    }
    ////////////

    public static void main(String[] args) {
        launch(args);
    }

}
