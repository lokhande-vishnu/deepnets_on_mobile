import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ['final_result'])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return

def main():
    Image = tf.placeholder(tf.float32, shape=[1,224,224,3], name='input') # input
    I = tf.reshape(Image, [1, 224*224*3])
    W = tf.Variable(tf.random_normal(shape=[224*224*3,5]), dtype=tf.float32, name='W') # weights
    b = tf.Variable(tf.random_normal(shape=[5]), dtype=tf.float32, name='b') # biases
    O = tf.nn.softmax(tf.matmul(I, W) + b, name='final_result') # activation / output
    
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        # save the graph
        # tf.train.write_graph(sess.graph_def, '.', 'tfdroid.pbtxt')  

        # normally you would do some training here
        # but fornow we will just assign something to W
        #sess.run(tf.assign(W, [[1, 2],[4,5],[7,8]]))
        #sess.run(tf.assign(b, [1,1]))
        init = tf.global_variables_initializer()
        sess.run(init)
        save_graph_to_file(sess, sess.graph, 'nn_forward.pb')
        
        '''
        #save a checkpoint file, which will store the above assignment  
        saver.save(sess, 'tfdroid.ckpt')

        
        MODEL_NAME = 'tfdroid'
    
        # Freeze the graph

        input_graph_path = MODEL_NAME+'.pbtxt'
        checkpoint_path = './'+MODEL_NAME+'.ckpt'
        input_saver_def_path = ""
        input_binary = False
        output_node_names = "final_result"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
        output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
        clear_devices = True

        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  input_binary, checkpoint_path, output_node_names,
                                  restore_op_name, filename_tensor_name,
                                  output_frozen_graph_name, clear_devices, "")
        
        f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
        f.write(output_graph_def.SerializeToString())
        '''
        
if __name__ == '__main__':
    main()
    
