<<<<<<< HEAD
# Overview

This repo contains code for the "TensorFlow for poets 2" series of codelabs.

There are multiple versions of this codelab depending on which version 
of the tensorflow libraries you plan on using:

* For [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) the new, ground up rewrite targeted at mobile devices
  use [this version of the codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite) 
* For the more mature [TensorFlow Mobile](https://www.tensorflow.org/mobile/mobile_intro) use 
  [this version of the codealab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2).


This repo contains simplified and trimmed down version of tensorflow's example image classification apps.

* The TensorFlow Lite version, in `android/tflite`, comes from [tensorflow/contrib/lite/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite).
* The Tensorflow Mobile version, in `android/tfmobile`, comes from [tensorflow/examples/android/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android).

The `scripts` directory contains helpers for the codelab. Some of these come from the main TensorFlow repository, and are included here so you can use them without also downloading the main TensorFlow repo (they are not part of the TensorFlow `pip` installation).

Please find the report here https://docs.google.com/document/d/1ob_Nf1P4cxuNwKlOEvdprDS3ZNa7RPhuipQqGuXSm64/edit
=======
# deepnets_on_mobile
CS744 Big Data Systems Course Project.
Please find the report here https://docs.google.com/document/d/1ob_Nf1P4cxuNwKlOEvdprDS3ZNa7RPhuipQqGuXSm64/edit?usp=sharing

## Clone Tensor Flow
git clone https://github.com/tensorflow/tensorflow.git

cd tensorflow/models/research/slim

## Generate Resnets model architecture
python export_inference_graph.py \
--alsologtostderr \
--model_name=resnets_v2_50 \
--output_file=/tmp/models/resnets_v2_50.pb

## Merge the above generated pb file with checkpoint file.
### You can download checkpoint file from https://github.com/tensorflow/models/tree/master/research/slim

bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/models/resnet_v2_50.pb \
--input_checkpoint=/tmp/checkpoints/resnet_v2_50.ckpt \
--input_binary=true --output_graph=/tmp/frozen_resnet_v2_50.pb \
--output_node_names=resnet_v2_50/predictions/Reshape_1

## For input and output node names use the following from tensor flow root directory
bazel run tensorflow/tools/graph_transforms:summarize_graph -- --in_graph=/tmp/models/resnet_v2_50.pb

## Optimize for inference if needed
cd tensorflow-for-poets-2

python -m tensorflow.python.tools.optimize_for_inference --input=\tmp\models\resnet_v2_50.pb --output=\tmp\models\optimized_resnet_v2_50.pb --input_names="input" --output_names="resnet_v2_50/predictions/Reshape_1"


## Quantize if needed
cd tensorflow-for-poets-2

python scripts/quantize_graph.py --input=/tmp/models/optimized_resnet_v2_50.pb --output=/tmp/models/quantized_resnet_v2_50.pb --output_node_names=resnet_v2_50/predictions/Reshape_1 --mode=eightbit
>>>>>>> f2686127dc9fec1d2113aa2948a1233b277334b4
