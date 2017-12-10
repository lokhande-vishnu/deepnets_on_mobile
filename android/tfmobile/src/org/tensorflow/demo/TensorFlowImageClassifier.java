/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import java.util.Random;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * A classifier specialized to label images using TensorFlow.
 */
public class TensorFlowImageClassifier implements Classifier {
    private static final String TAG = "TensorFlowImageClassifier";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;

    // FC config values
    private String inputfc;
    private String weight;
    private String bias;
    private String outputfc;
    private float[] outputfc_val;
    private float[] weight_new_val;
    private String true_output;
    private String weight_new;
    private String[] fc_outputs;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;
    private String[] backpropOutputs;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;
    private TensorFlowInferenceInterface fcInterface;

    private TensorFlowImageClassifier() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize     The input size. A square image of inputSize x inputSize is assumed.
     * @param imageMean     The assumed mean of the image values.
     * @param imageStd      The assumed std of the image values.
     * @param inputName     The label of the image input node.
     * @param outputName    The label of the output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            int imageMean,
            float imageStd,
            String inputName,
            String outputName,
            String fc_filename) {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        Log.i(TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                c.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = c.inferenceInterface.graphOperation(outputName);
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;
        c.imageMean = imageMean;
        c.imageStd = imageStd;

        // Pre-allocate buffers.
        c.outputNames = new String[]{outputName};
        c.intValues = new int[inputSize * inputSize];
        c.floatValues = new float[inputSize * inputSize * 3];
        c.outputs = new float[numClasses];

        // FC variables
        c.inputfc = "inputfc";
        c.weight = "weight";
        c.bias = "bias";
        c.outputfc = "outputfc";
        c.outputfc_val = new float[numClasses];
        c.true_output = "true_output";
        c.weight_new = "weight_new";
        c.weight_new_val = new float[10 * numClasses];
        c.fc_outputs = new String[]{"outputfc", "weight_new"};
        c.fcInterface = new TensorFlowInferenceInterface(assetManager, fc_filename);

        return c;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();

        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > THRESHOLD) {
                pq.add(
                        new Recognition(
                                "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
            }
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"


        // Fully connected layer

        float[] inputfc_var = new float[10];
        for (int i = 0; i < inputfc_var.length; i++) {
            inputfc_var[i] = (float) Math.random();
        }
        float[] weight_var = new float[50];
        for (int i = 0; i < weight_var.length; i++) {
            weight_var[i] = (float) Math.random();
        }
        float[] bias_var = new float[5];
        for (int i = 0; i < bias_var.length; i++) {
            bias_var[i] = (float) Math.random();
        }
        float[] toutput_var = new float[5];
        for (int i = 0; i < toutput_var.length; i++) {
            toutput_var[i] = (float) Math.random();
        }

        // Feed data

        Trace.beginSection("feed");
        fcInterface.feed(inputfc, inputfc_var, 1, 10);
        Trace.endSection();

        Trace.beginSection("feed");
        fcInterface.feed(weight, weight_var, 10, 5);
        Trace.endSection();

        Trace.beginSection("feed");
        fcInterface.feed(bias, bias_var, 5);
        Trace.endSection();

        Trace.beginSection("feed");
        fcInterface.feed(true_output, toutput_var, 5);
        Trace.endSection();

        // Run the FC network
        Trace.beginSection("run");
        fcInterface.run(fc_outputs, logStats);
        Trace.endSection();

        // Get the outputs and new weights
        Trace.beginSection("fetch");
        fcInterface.fetch(outputfc, outputfc_val);
        Trace.endSection();

        Trace.beginSection("fetch");
        fcInterface.fetch(weight_new, weight_new_val);
        Trace.endSection();


    /*

    // Run backprop
    Trace.beginSection("run");
    backpropInterface.run(backpropOutputs, logStats);
    Trace.endSection();
    */


        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
