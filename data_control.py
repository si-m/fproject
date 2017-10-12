import numpy as np

class DataControl(object):
  def __init__(self, data_dir, batch_size, test_size=0.2):
    """
    """
    self._data_dir = data_dir
    self._batch_size   = batch_size
    self._train_tweets = []
    self._train_labels = []
    self._self._test_labels = []
    self._test_tweets  = []
    self._test_steps   = 0
    self._train_steps  = 0
    self.__load_data()

  def __load_data(self):
    #load train set
    self._train_tweets = np.load(self._data_dir + 'train_vec_tweets.npy')
    self._train_labels = np.load(self._data_dir + 'train_vec_labels.npy')
    #load test set
    self._test_tweets = np.load(self._data_dir + 'test_vec_tweets.npy')
    self._test_labels = np.load(self._data_dir + 'test_vec_tweets.npy')

    self._test_steps  = int(len(self._test_tweets)/self._batch_size)
    self._train_steps = int(len(self._train_tweets)/self._batch_size)

  def next_train_batch_for_step(self, step):
    """
    """
    offset = (step * self._batch_size) % (len(self._train_tweets) - self._batch_size)
    batch_tweets = self._train_tweets[offset : (offset + self._batch_size)]
    batch_labels = self._train_labels[offset : (offset + self._batch_size)]
    return batch_tweets, batch_labels

  def train_steps_per_epoch():
    return self._train_steps

  def test_steps_per_epoch():
    return self._test_steps

  def next_test_batch_for_step(self, step):
    """
    """
    test_offset = (step * self._batch_size) % (len(self._test_tweets) - self._batch_size)
    test_batch_tweets = self._test_tweets[test_offset : (test_offset + self._batch_size)]
    test_batch_labels = self._test_labels[test_offset : (test_offset + self._batch_size)]

    return test_batch_tweets, test_batch_labels