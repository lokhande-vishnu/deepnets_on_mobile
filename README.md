# deepnets_on_mobile

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
