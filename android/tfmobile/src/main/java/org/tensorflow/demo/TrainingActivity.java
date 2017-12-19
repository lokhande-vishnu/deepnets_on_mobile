package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.AsyncTask;
import android.os.Bundle;
import android.app.Activity;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;


import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.Logger;

import java.io.File;
import java.util.List;

public class TrainingActivity extends Activity {
    private static final Logger LOGGER = new Logger();

    // These are the settings for the original v1 Inception model. If you want to
    // use a model that's been produced from the TensorFlow for Poets codelab,
    // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
    // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
    // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
    // the ones you produced.
    //
    // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
    // model first:
    //
    // python strip_unused.py \
    // --input_graph=<retrained-pb-file> \
    // --output_graph=<your-stripped-pb-file> \
    // --input_node_names="Mul" \
    // --output_node_names="final_result" \
    // --input_binary=true

  /* Inception V3
  private static final int INPUT_SIZE = 299;
  private static final int IMAGE_MEAN = 128;
  private static final float IMAGE_STD = 128.0f;
  private static final String INPUT_NAME = "Mul:0";
  private static final String OUTPUT_NAME = "final_result";
  */

    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    // private static final String INPUT_NAME = "input";
    // private static final String OUTPUT_NAME = "MobilenetV1/Predictions/Softmax";
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "MobilenetV1/Predictions/Reshape";

    private static final String MODEL_FILE = "file:///android_asset/store_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/labels.txt";
    private static final String FC_FILE = "file:///android_asset/update_fc.pb";

    private static final boolean SAVE_PREVIEW_BITMAP = false;

    private static final boolean MAINTAIN_ASPECT = true;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private Classifier classifier;

    private Integer sensorOrientation;

    private int previewWidth = 0;
    private int previewHeight = 0;
    private byte[][] yuvBytes;
    private int[] rgbBytes = null;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;

    private Bitmap cropCopyBitmap;

    private boolean computing = false;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private ResultsView resultsView;

    private BorderedText borderedText;

    private long lastProcessingTimeMs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_training);
        Log.d(TrainingActivity.class.getSimpleName(), "Test Log");
        classifier = TensorFlowImageClassifier.create(
                getAssets(),
                MODEL_FILE,
                LABEL_FILE,
                INPUT_SIZE,
                IMAGE_MEAN,
                IMAGE_STD,
                INPUT_NAME,
                OUTPUT_NAME,
                FC_FILE);
        File dir = new File(Environment.getExternalStorageDirectory() + "/images/");
        new Task().execute(dir);


    }

    private class Task extends AsyncTask<File, Integer, String> {

        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            // Do something like display a progress bar
        }

        // This is run in a background thread
        @Override
        protected String doInBackground(File... params) {
            int totEpochs = 10;
            for (int ep = 0; ep < totEpochs; ep++) {
                File dir = params[0];
                // get the string from params, which is an array
                File[] directoryListing = dir.listFiles();
                if (directoryListing != null) {
                    for (File child : directoryListing) {
                        if (child.isDirectory()) {
                            String childName = child.getName();
                            System.out.println("Directory: " + childName);
                            File[] imageLists = child.listFiles();
                            float[] toutput_var;
                            if (childName.equalsIgnoreCase("daisy")) {
                                toutput_var = new float[]{1, 0, 0, 0, 0};
                            } else if (childName.equalsIgnoreCase("dandelion")) {
                                toutput_var = new float[]{0, 1, 0, 0, 0};
                            } else if (childName.equalsIgnoreCase("roses")) {
                                toutput_var = new float[]{0, 0, 1, 0, 0};
                            } else if (childName.equalsIgnoreCase("sunflowers")) {
                                toutput_var = new float[]{0, 0, 0, 1, 0};
                            } else {
                                toutput_var = new float[]{0, 0, 0, 0, 1};
                            }

                            for (File image : imageLists) {
                                String filePath = image.getPath();
                                Bitmap bitmap = BitmapFactory.decodeFile(filePath);
                                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
                                final long startTime = SystemClock.uptimeMillis();
                                final Classifier.MyResult myresult = classifier.recognizeImage(resizedBitmap, toutput_var);
                                final List<Classifier.Recognition> results = myresult.getFirst();
                                final float error = myresult.getSecond();
                                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                                Log.d(TrainingActivity.class.getSimpleName(), "time=" + lastProcessingTimeMs);
                            }
                        }
                    }
                } else {
                    // Handle the case where dir is not really a directory.
                    // Checking dir.isDirectory() above would not be sufficient
                    // to avoid race conditions with another process that deletes
                    // directories.
                    System.out.println("Something wrong");
                }
            }

            return "this string is passed to onPostExecute";
        }

        // This is called from background thread but runs in UI
        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);

            // Do things like update the progress bar
        }

        // This runs in UI when background thread finishes
        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);

            // Do things like hide the progress bar or change a TextView
        }
    }
}
