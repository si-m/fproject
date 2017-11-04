import tensorflow as tf
import numpy as np

tf.reset_default_graph()

export_dir = "models-binary/"

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('checkpoints/pretrained_bin_lstm.ckpt-3750.meta')
saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))


graph = tf.get_default_graph()

tweets = graph.get_tensor_by_name("tweets:0")
prediction = graph.get_tensor_by_name("pred_op:0")

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

inputs  = tf.saved_model.utils.build_tensor_info(tweets)
outputs = tf.saved_model.utils.build_tensor_info(prediction)

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

prediction_signature = (
tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'tweets': inputs},
    outputs={'scores': outputs},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
    },legacy_init_op=legacy_init_op)


builder.save()