/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.sudheerherle.ObjectRecognition;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 *
 * @author I14746
 */
public class RecognizerController {
    
    
    private byte[] graphDef;
    private List<String> labels;
    private boolean modelselected = false;
    private String imagepath;
    
    @FXML
    private Label predictionLabel;
    @FXML
    private ProgressBar predcitionBar;
    @FXML
    private Button BrowseImage;
    @FXML
    private Button BrowseInception;
    @FXML
    private TextField InceptionPathTxt;
    @FXML
    private ImageView imgview;
    @FXML
    private void BrowseInception(){
        DirectoryChooser chooser = new DirectoryChooser();
        chooser.setTitle("Hex file selection");
        chooser.setInitialDirectory(new File(System.getProperty("user.home")));
        File selectedF = chooser.showDialog(BrowseInception.getParent().getScene().getWindow());
        
        if(selectedF!=null){
            modelselected = true;
            graphDef = readAllBytesOrExit(Paths.get(selectedF.getAbsolutePath(), "tensorflow_inception_graph.pb"));
            labels = readAllLinesOrExit(Paths.get(selectedF.getAbsolutePath(), "imagenet_comp_graph_label_strings.txt"));
            InceptionPathTxt.setText(selectedF.getAbsolutePath());
        }
    }
    
    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }

    @FXML
    private void BrowseImage(){
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Hex file selection");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("JPEG File", "*.JPG", "*.JPEG"));
        chooser.setInitialDirectory(new File(System.getProperty("user.home")));
        File selectedF = chooser.showOpenDialog(BrowseImage.getParent().getScene().getWindow());
        
        if(selectedF!=null){
            this.imagepath = selectedF.getAbsolutePath();
            String imagepath = null;
                    try {
                        imagepath = selectedF.toURI().toURL().toString();
                    } catch (MalformedURLException ex) {
                        Logger.getLogger(RecognizerController.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    Image image = new Image(imagepath);
                    imgview.setFitHeight(imgview.getFitHeight());
                    imgview.setFitWidth(imgview.getFitWidth());
                    imgview.setImage(image);
                    Predict();
        }

    }
    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                    Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }
    
     private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }
     
    @FXML
    private void Predict(){
        predictionLabel.setText("Prediction:\t Please wait...");
        predcitionBar.setProgress(ProgressBar.INDETERMINATE_PROGRESS);
         Task t1 = new Task() {
                @Override
                protected Object call() throws Exception {
         byte[] imageBytes = readAllBytesOrExit(Paths.get(imagepath));
//
            try (Tensor image = Tensor.create(imageBytes)) {
                float[] labelProbabilities = executeInceptionGraph(graphDef, image);
                int bestLabelIdx = maxIndex(labelProbabilities);
                 Platform.runLater(() -> {
                     predcitionBar.setProgress(labelProbabilities[bestLabelIdx]);
                predictionLabel.setText("");
                predictionLabel.setText(String.format(
                                "Prediction:\t %s (%.2f%% likely)",
                                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
            });
               
                System.out.println(
                        String.format(
                                "BEST MATCH: %s (%.2f%% likely)",
                                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
            }
                    return null;

        }
         };
         Thread th = new Thread(t1);
         th.start();
    
}
}
