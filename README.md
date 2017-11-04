# Final Project Informatic Engineering

This is an implementation of a Recurrent Neural Network for sentiment analysis.

## Getting Started

### Pre-requisites
* python3
* numpy
* scipy
* gensim
* json-tricks
* nltk
* docker

#### Download spanish word2vec model:

```sh
$ wget ‐‐directory-prefix=data_process/ http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz
$ cd data_process 
$ gunzip SBW-vectors-300-min5.bin.gz
```

### Building Dataset

Creating the test and training sets:

```sh
$ cd data_process/
$ python3 build_dataset.py -i full_spanish_dataset.json 
```

### Vectorizing dataset

```sh
$ python3 vectorize.py -i pre_train_set.json -o train
$ python3 vectorize.py -i pre_test_set.json -o test
```

Move the final vectorized sets to '/data_es':

```
$ mv test_vec_labels.npy train_vec_tweets.npy test_vec_tweets.npy train_vec_labels.npy ../data_es/
```

## Training
### With tensorflow in local machine

```sh
$ python3 sentiment_rnn.py
```
### With Docker
This docker image has all the dependencies for tensoflow and tensorboard.
You could start a shell in the cointainer or just use the jupyter notebook:
#### Shell
```sh
$ docker run -it --rm -v /path/to/this/repo:/home/jovyan/work -p 8888:8888 jupyter/tensorflow-notebook:latest bash
```
In the container:
```sh
jovyan@2261d9443deb:~$ cd work/fproject
jovyan@2261d9443deb:~$ python sentiment_rnn.py
```
#### Jupyter notebook
```sh
$ docker run -it --rm -v /path/to/this/repo:/home/jovyan/work -p 8888:8888 jupyter/tensorflow-notebook:latest
```

## Tensorboard
```sh
$ tensorboard --logdir /path/to/log/folder
```
In the container:
```sh
jovyan@2261d9443deb:~$ cd work/fpie/tensorboard
jovyan@2261d9443deb:~$ tensorboard --logdir log/folder
```

## Exporting Model

After trainig we need to export the model.

We have to cd into the export folder and change the checkpoint path to our last checkpoint.
```sh
$ cd export
$ python save_model.py
```

## Serving

The last step is to serve our model.
Go To -https://github.com/si-m/fproject_serv

## Author

* **Santiago IM** - *Initial work* - si.musielack@gmail.com

## Info

* The Spanish Billion Words Corpus and Embeddings linguistic resource. http://crscardellino.me/SBWCE/
* Understanding LSTM RNN. -http://colah.github.io/posts/2015-08-Understanding-LSTMs/
* Rnn Effectiveness. -http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* Gensim Word2vec. - https://radimrehurek.com/gensim/models/word2vec.html
* Learning rate. - https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
* Docker image. -https://github.com/jupyter/docker-stacks/tree/master/tensorflow-notebook
* Dropout LSTM google research. -https://arxiv.org/pdf/1603.05118.pdf

## ToDo

* Add an embedding layer using the gensim word2vec model
* Increase the dataset to improve the rnn loss and accuracy
* Start another branch to continue with the 3 classes sentiment analysis.