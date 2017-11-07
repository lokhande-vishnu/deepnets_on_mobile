package org.tensorflow.demo;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

public class BenchmarkingActivity extends Activity implements AdapterView.OnItemSelectedListener{

    private static final String[] tensorflow_models = new String[]{"tensorflow_inception_graph.pb",
            "mobilenet.pb", "optimized_resnet_v2_50.pb", "quantized_resnet_v2_50.pb"};
    public static final String MODEL_FOLDER = "file:///android_asset/";
    public static final String IMAGE_FOLDER = "file:///android_asset/images";
    int selectedModelIndex = 0;

    /* *** UI Elements ****/
    Spinner tensorflowModelsSpinner;
    Button runBenchmarkButton;
    TextView progressReportTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_benchmarking);

        tensorflowModelsSpinner = (Spinner) findViewById(R.id.tensorflow_models_list);
        runBenchmarkButton = (Button) findViewById(R.id.run_benchmark);
        progressReportTextView = (TextView) findViewById(R.id.progress_report);

        ArrayAdapter<String> adapter = new ArrayAdapter<String>(BenchmarkingActivity.this,
                android.R.layout.simple_spinner_item, tensorflow_models);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        tensorflowModelsSpinner.setAdapter(adapter);
        tensorflowModelsSpinner.setOnItemSelectedListener(this);

        runBenchmarkButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String model = tensorflow_models[selectedModelIndex];
                String modelPath = MODEL_FOLDER + model;
            }
        });
    }

    public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {
        selectedModelIndex = position;
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }
}
