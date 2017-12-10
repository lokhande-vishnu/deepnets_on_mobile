import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ['outputfc', 'weight_new'])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return

def main():
    input = tf.placeholder(tf.float32, shape=[1,1001], name='inputfc') # input
    weight = tf.placeholder(tf.float32, shape=[1001,5], name='weight') # weight
    bias = tf.placeholder(tf.float32, shape=[5], name='bias') # bias
    output = tf.nn.softmax(tf.matmul(input, weight) + bias, name='outputfc') # activation / output

    true_output = tf.placeholder(tf.float32, shape = [5], name = 'true_output')
    delta = tf.reshape(true_output - output, [5,1])
    grad_updates = tf.matmul(input, delta, transpose_a = True, transpose_b = True)
    weight_new = tf.subtract(weight, 0.01 * grad_updates, name='weight_new')
    
    #saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)
        save_graph_to_file(sess, sess.graph, 'update_fc.pb')
        
if __name__ == '__main__':
    main()
    
