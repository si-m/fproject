import tensorflow as tf
import numpy as np
import math
import datetime
import os
import random

# This is the sentiment analisys RNN to predict tweets sentiments in this 2 classes (POSITIVE, NEGATIVE) 

# set variables
num_epochs = 15
tweet_size = 20
hidden_size = 500
vec_size = 300
batch_size = 512
number_of_layers= 2
number_of_classes= 1
starter_learning_rate = 0.001

tf.reset_default_graph()

# Create a session 
session = tf.Session()

# Inputs placeholders
tweets = tf.placeholder(tf.float32, [None, tweet_size, vec_size], "tweets")
labels = tf.placeholder(tf.float32, [None], "labels")

# Placeholder for dropout
keep_prob = tf.placeholder_with_default(1.0,[], name="keep_prob") 

# make the lstm cells, and wrap them in MultiRNNCell for multiple layers
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    return tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)

multi_lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)], state_is_tuple=True)

batch_size_T  = tf.shape(tweets)[0]
zerostate = multi_lstm_cells.zero_state(batch_size_T, dtype=tf.float32)

# Creates a recurrent neural network
_, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, tweets, dtype=tf.float32, initial_state=zerostate)

sentiments = tf.contrib.layers.fully_connected(final_state[-1][-1], num_outputs=number_of_classes, activation_fn=None, weights_initializer=tf.random_normal_initializer(),
  biases_initializer=tf.random_normal_initializer(), scope="fully_connected")

sentiments = tf.squeeze(sentiments, [1])

predictions = tf.nn.sigmoid(sentiments, name="pred_op")

with tf.name_scope("loss"):
    # define cross entropy loss function
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=sentiments, labels=labels)
    loss = tf.reduce_mean(losses, name="loss_op")
    #tensorboard summaries
    tf.summary.scalar("loss", loss)


with tf.name_scope("accuracy"):
    # round our actual probabilities to compute error
    accuracy = tf.to_float(tf.equal(tf.to_float(tf.greater_equal(predictions, 0.5)), labels))
    accuracy = tf.reduce_mean(tf.cast(accuracy, dtype=tf.float32), name="acc_op")
    #tensorboard summaries
    tf.summary.scalar("accuracy", accuracy)
    
    
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=True)    
# define our optimizer to minimize the loss
with tf.name_scope("train"):
    optimizer = tf.train. AdamOptimizer(learning_rate).minimize(loss)

#tensorboard summaries
merged_summary = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
train_writer = tf.summary.FileWriter(logdir + "-training", session.graph)
test_writer  = tf.summary.FileWriter(logdir + "-testing", session.graph)

# initialize any variables
tf.global_variables_initializer().run(session=session)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# load our data and separate it into tweets and labels
train_tweets = np.load('data_es/train_vec_tweets.npy')
train_labels = np.load('data_es/train_vec_labels.npy')

test_tweets = np.load('data_es/test_vec_tweets.npy')
test_labels = np.load('data_es/test_vec_labels.npy')



steps = int(len(train_tweets)/batch_size)
summary_step = 0

for epoch in range(num_epochs):
    for step in range(steps):

        offset = (step * batch_size) % (len(train_tweets) - batch_size)
        batch_tweets = train_tweets[offset : (offset + batch_size)]
        batch_labels = train_labels[offset : (offset + batch_size)]

        data = {tweets: batch_tweets, labels: batch_labels, keep_prob: 0.8}

        #run operations in graph
        _, loss_train, accuracy_train = session.run([optimizer, loss, accuracy], feed_dict=data)
    
        if (step % 50 == 0):
            test_loss = []
            test_accuracy = []
            print("Epoch:", epoch, "Step:", step)
            print("Summary step:",(summary_step))
            print("Train loss:", loss_train)
            print("Train accuracy:%.3f%%" % (accuracy_train*100))
            
            #tensorboard visualizations
            summary = session.run(merged_summary, feed_dict=data)
            train_writer.add_summary(summary, summary_step)
            
            for batch_num in range(int(len(test_tweets)/batch_size)):
                
                test_offset = (batch_num * batch_size) % (len(test_tweets) - batch_size)
                test_batch_tweets = test_tweets[test_offset : (test_offset + batch_size)]
                test_batch_labels = test_labels[test_offset : (test_offset + batch_size)]

                data_testing = {tweets: test_batch_tweets, labels: test_batch_labels, keep_prob: 1.0}

                loss_test, accuracy_test = session.run([loss, accuracy], feed_dict=data_testing)

                test_loss.append(loss_test)
                test_accuracy.append(accuracy_test)
            
            
            test_offset = (random.randint(0,len(test_tweets) - batch_size) * batch_size) % (len(test_tweets) - batch_size)
            test_batch_tweets = test_tweets[test_offset : (test_offset + batch_size)]
            test_batch_labels = test_labels[test_offset : (test_offset + batch_size)]
            
            data_testing = {tweets: test_batch_tweets, labels: test_batch_labels, keep_prob: 1.0}

            #tensorboard visualizations
            test_summary = session.run(merged_summary, feed_dict=data_testing)
            test_writer.add_summary(test_summary, summary_step)
            
            summary_step += 50
            global_step += 50
            
            print("Test loss:%.3f" % np.mean(test_loss))
            print("Test accuracy:%.3f%%" % (np.mean(test_accuracy)*100))


saver.save(session, 'checkpoints/pretrained_bin_lstm.ckpt', global_step=summary_step)