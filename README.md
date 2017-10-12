# Final Project Informatic Engineering

This is an implementation of a Recurrent Neural Network for sentiment analysis.

## Getting Started

### Prerequisites
* python3
* numpy
* scipy
* gensim
* json-tricks
* nltk

Download spanish word2vec model:

```
wget ‐‐directory-prefix=data_process/ http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz
cd data_process 
gunzip SBW-vectors-300-min5.bin.gz
```

### Building Dataset

Creating the test and training sets:

```
cd data_process/
python build_dataset.py -i full_spanish_dataset.json 
```

### Vectorizing dataset

```
python vectorize.py -i pre_train_set.json -o train
python vectorize.py -i pre_test_set.json -o test
```

Move the final vectorized sets to '/data_es':

```
mv test_vec_labels.npy train_vec_tweets.npy test_vec_tweets.npy train_vec_labels.npy ../data_es/
```

### Training the network
```
python sentiment_rnn.py
```

## Author

* **Santiago IM** - *Initial work* -

## Contributions

* **Murphy**


## Acknowledgments

* The Spanish Billion Words Corpus and Embeddings linguistic resource. http://crscardellino.me/SBWCE/
* Understanding LSTM RNN. -http://colah.github.io/posts/2015-08-Understanding-LSTMs/
* Rnn Effectiveness. -http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* Gensim Word2vec. - https://radimrehurek.com/gensim/models/word2vec.html
* Learning rate. - https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
