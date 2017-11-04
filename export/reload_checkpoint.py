import tensorflow as tf
import numpy as np
from data_process import utils

tf.reset_default_graph()

sess = tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('checkpoints/pretrained_bin_lstm.ckpt-3750.meta')
saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

# print(sess.run('saved:0'))
# This will print 2, which is the value of bias that we saved

graph = tf.get_default_graph()
    
# Access saved Variables directly
tweets = graph.get_tensor_by_name("tweets:0")
# labels = graph.get_tensor_by_name("labels:0")
# keep_prob = graph.get_tensor_by_name("keep_prob:0")

# [print(tensor.name) for tensor in tf.get_default_graph().as_graph_def().node]

# #Now, access the op that you want to run. 
pred = graph.get_tensor_by_name("pred_op:0")
# loss = graph.get_tensor_by_name("loss/loss_op:0")
# acc  = graph.get_tensor_by_name("accuracy/acc_op:0")


data = {tweets: utils.tweetsToVec(["Gran día para pasear en bici por Ka ciudad!","Esto se complica...","Me encanta la tarde","Se me acabaron las vacaciones","la puta madrea tengo un examen","Muy bueno el cumple de Feli"])}

pred_ = sess.run([pred], feed_dict=data)

print("Prediction:", pred_)