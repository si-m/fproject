import tensorflow as tf
import numpy as np
import math
import datetime
import os

# tf.flags.DEFINE_integer('train_steps', 1000,
#                         'Number of training steps')

# set variables
num_epochs = 15
tweet_size = 20
hidden_size = 200
vec_size = 300
batch_size = 512
number_of_layers= 1
number_of_classes= 3
learning_rate = 0.001

TRAIN_DIR="/checkpoints"

# this just makes sure that all our following operations will be placed in the right graph.
tf.reset_default_graph()

# create a session variable that we can run later.
session = tf.Session()

# make placeholders for data we'll feed in
tweets = tf.placeholder(tf.float32, [None, tweet_size, vec_size], "tweets")
labels = tf.placeholder(tf.float32, [None, number_of_classes], "labels")

# placeholder for dropout
keep_prob = tf.placeholder(tf.float32) 

# make the lstm cells, and wrap them in MultiRNNCell for multiple layers
def lstm_cell():
  cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
  return tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)

multi_lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)], state_is_tuple=True)

#Creates a recurrent neural network
output, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, tweets, dtype=tf.float32)

with tf.name_scope("final_layer"):
  #weight and bias to shape the final layer
  W = tf.get_variable("weight_matrix", [hidden_size, number_of_classes], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(hidden_size)))
  b = tf.get_variable("bias", [number_of_classes], initializer=tf.constant_initializer(1.0))

  sentiments = tf.matmul(final_state[-1][-1], W) + b

prob = tf.nn.softmax(sentiments)
tf.summary.histogram('softmax', prob)

with tf.name_scope("loss"):
  # define cross entropy loss function
  losses = tf.nn.softmax_cross_entropy_with_logits(logits=sentiments, labels=labels)
  loss = tf.reduce_mean(losses)
  tf.summary.scalar("loss", loss)

with tf.name_scope("accuracy"):
  # round our actual probabilities to compute error
  accuracy = tf.to_float(tf.equal(tf.argmax(prob,1), tf.argmax(labels,1)))
  accuracy = tf.reduce_mean(tf.cast(accuracy, dtype=tf.float32))
  tf.summary.scalar("accuracy", accuracy)

# define our optimizer to minimize the loss
with tf.name_scope("train"):
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#tensorboard summaries
merged_summary = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, session.graph)

# initialize any variables
tf.global_variables_initializer().run(session=session)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# load our data and separate it into tweets and labels
train_tweets = np.load('data_es/train_vec_tweets.npy')
train_labels = np.load('data_es/train_vec_labels.npy')

test_tweets = np.load('data_es/test_vec_tweets.npy')
test_labels = np.load('data_es/test_vec_labels.npy')

test_tweets = test_tweets[:9000]
test_labels = test_labels[:9000]

steps = int(len(train_tweets)/batch_size)
for epoch in range(num_epochs):
  for step in range(steps):

    offset = (step * batch_size) % (len(train_tweets) - batch_size)
    batch_tweets = train_tweets[offset : (offset + batch_size)]
    batch_labels = train_labels[offset : (offset + batch_size)]

    data = {tweets: batch_tweets, labels: batch_labels, keep_prob: 0.75}

    #run operations in graph
    _, loss_train, accuracy_train, _final_state, _output = session.run([optimizer, loss, accuracy, final_state, output], feed_dict=data)

    if (step % 50 == 0):
      test_loss = []
      test_accuracy = []
      print("Epoch:", epoch, "Step:", step)
      print("Train loss:", loss_train)
      print("Train accuracy:%.3f%%" % (accuracy_train*100))

      summary = session.run(merged_summary, feed_dict=data)
      print("Summary steps:",(step+(epoch*steps)))
      writer.add_summary(summary,(step+(epoch*steps)))

      for batch_num in range(int(len(test_tweets)/batch_size)):
        test_offset = (batch_num * batch_size) % (len(test_tweets) - batch_size)
        test_batch_tweets = test_tweets[test_offset : (test_offset + batch_size)]
        test_batch_labels = test_labels[test_offset : (test_offset + batch_size)]

        data_testing = {tweets: test_batch_tweets, labels: test_batch_labels, keep_prob: 1.0}

        loss_test, accuracy_test = session.run([loss, accuracy], feed_dict=data_testing)

        test_loss.append(loss_test)
        test_accuracy.append(accuracy_test)

      print("Test loss:%.3f" % np.mean(test_loss))
      print("Test accuracy:%.3f%%" % (np.mean(test_accuracy)*100))


  saver.save(session, 'checkpoints/pretrained_lstm.ckpt', global_step=step)