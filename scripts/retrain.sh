python retrain.py\
       --image_dir='/Users/lokhande/images'\
       --output_graph=retrain_summaries/graph.pb\
       --summaries_dir=retrain_summaries\
       --how_many_training_steps=500\
       --learning_rate=0.01\
       --train_batch_size=1\
       --architecture='mobilenet_1.0_224'
