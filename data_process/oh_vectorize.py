import sys, getopt
import gensim
import numpy as np
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments

def one_hot(raw_data, vocab_size):
    data = np.zeros((len(raw_data), 20, vocab_size))
    for tweet_index in range(len(raw_data)):
        tweet = raw_data[tweet_index]
        for word_index in range(20):
            word_id = tweet[word_index]
            data[tweet_index, word_index, word_id] = 1
    return data

def main(argv):
	inputfile = ''
	trainfile = 'pre_train_set.json'
	testfile  = 'pre_test_set.json'
	filtered_set  = []
	train_set = []
	test_set = []
	try:
		opts, args = getopt.getopt(argv,"hi:",["ifile="])
	except getopt.GetoptError:
		print('build_set.py -i <inputfile.json>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('oh_vectorize.py -i <inputfile.json>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg

	uniq = set()
	vocab = set()
	tknzr = TweetTokenizer()

	with open(inputfile, 'r') as fpt:
		l = fpt.readlines()
	for row, d in enumerate(l):
		try:
			raw = json.loads(str(d, encoding='utf-8'))
		except TypeError:
			raw = json.loads(d)
		if(raw['klass'] != 'NONE' and (raw['id'] not in uniq)):
			filtered_set.append({'text': raw['text'], 'klass': raw['klass']})
			uniq.add(raw['id'])
			vocab.add(tknzr.tokenize(raw['text'])).flatten

	with open(trainfile, 'w') as outfile:
			json_data = dumps(train_set)
			outfile.write(json_data)
			outfile.close()

if __name__ == "__main__":
   main(sys.argv[1:])