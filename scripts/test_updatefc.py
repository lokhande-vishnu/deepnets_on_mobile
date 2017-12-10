# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

if __name__ == "__main__":
  model_file = "update_fc.pb"
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", help="graph/model to be executed")
  args = parser.parse_args()
  if args.graph:
    model_file = args.graph

  graph = load_graph(model_file)
  input_opt = graph.get_operation_by_name('import/inputfc');
  weight_opt = graph.get_operation_by_name('import/weight');
  bias_opt = graph.get_operation_by_name('import/bias');
  output_opt = graph.get_operation_by_name('import/outputfc');
  toutput_opt = graph.get_operation_by_name('import/true_output');
  nweight_opt = graph.get_operation_by_name('import/weight_new');
  
  input_var = np.random.rand(1,1001)
  weight_var = np.random.rand(1001,5)
  bias_var = np.random.rand(5)
  toutput_var = np.random.rand(5)
      
  with tf.Session(graph=graph) as sess:
    start = time.time()
    out, nwei = sess.run([output_opt.outputs[0], nweight_opt.outputs[0]],
                      {input_opt.outputs[0]: input_var,
                       weight_opt.outputs[0]: weight_var,
                       bias_opt.outputs[0]: bias_var,
                       toutput_opt.outputs[0]: toutput_var})
    end=time.time()

  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
  print('out', len(out), len(out[0]))
  print('nwei', len(nwei), len(nwei[0]))
  
