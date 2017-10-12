#!/usr/bin/python
import sys, getopt
import gensim
import numpy as np
import json
from json_tricks.np import dump, dumps, load
from nltk.tokenize import TweetTokenizer
import random

# number of words in a tweet
MAX_NB_WORDS=20

def name():
	names = ['Juan','Pedro','Luis','Adrián','Carlos','Cristina','Marta','Sara','María','Lucía','Paula','Laura']
	return random.choice(names)

def words_to_dicc(words, vocab_size):
	ind = 0
	dicc = {}
	for w in words:
		dicc[w] = ind
		ind+=1
	return dicc

def label_to_value(label):
	table = {"P":[1,0,0],"positive":[1,0,0], "NEU":[0,1,0], "neutral":[0,1,0], "N":[0,0,1], "negative":[0,0,1]}
	return table[label]

def main(argv):
	inputfile = ''
	outputfile = ''

	tweets = []
	labels = []
	tknzr = TweetTokenizer()

	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print('vectorize.py -i <inputfile.json> -o <outputfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('vectorize.py -i <inputfile.json> -o <outputfile>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg


	data = load(open(inputfile, 'r'))

	# pre process data all uniques and with accepted klasses
	for pre_data in data:
	  if(pre_data['klass'] != 'NONE'):
	  	tweets.append( tknzr.tokenize(pre_data['text']) )
	  	labels.append( pre_data['klass'])
	  	
	# load http://crscardellino.me/SBWCE/ trained model
	model = gensim.models.KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin', binary=True)

	shape = (len(tweets), MAX_NB_WORDS, 300)
	tweets_tensor = np.zeros(shape, dtype=np.float32)

	for i in range(len(tweets)):
		#vectorizing each word in the tweet with a vector shape = (300,)
		for f in range(len(tweets[i])):
			word = tweets[i][f]
			if f >= MAX_NB_WORDS:
				continue
			#if is not in the vocabulary
			if word in model.wv.vocab:
				tweets_tensor[i][f] = model.wv[word]


	labels_array = np.array(list(map(lambda label: label_to_value(label), labels)), dtype=np.int32)

	np.save(outputfile + '_vec_tweets.npy',tweets_tensor)
	np.save(outputfile + '_vec_labels.npy',labels_array)


if __name__ == "__main__":
   main(sys.argv[1:])