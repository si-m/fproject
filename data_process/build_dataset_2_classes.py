#!/usr/bin/env python
import sys, getopt
import numpy as np
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments
import random

def main(argv):
	inputfile = ''
	trainfile = 'pre_train_set.json'
	testfile  = 'pre_test_set.json'
	positive_set, negative_set, export_set = [], [], []
	set_size = 0
	train_set, test_set = [], []
	uniq = set()

	try:
		opts, args = getopt.getopt(argv,"hi:",["ifile="])
	except getopt.GetoptError:
		print('build_set.py -i <inputfile.json>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('build_set.py -i <inputfile.json>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg

	with open(inputfile, 'r') as fpt:
		l = fpt.readlines()
	for row, d in enumerate(l):
		try:
			raw = json.loads(str(d, encoding='utf-8'))
		except TypeError:
			raw = json.loads(d)
		if(raw['id'] not in uniq):
			uniq.add(raw['id'])
			if raw['klass'] == "positive":
				positive_set.append({'text': raw['text'], 'klass': raw['klass']})
			if raw['klass'] == "negative":
				negative_set.append({'text': raw['text'], 'klass': raw['klass']})

	print("Sizes: ")
	# check sizes
	print("Uniques tweets: ",len(uniq))
	print("Positive tweets:", len(positive_set))
	print("Negative tweets:", len(negative_set))


	# smallest
	set_size = min(len(positive_set), len(negative_set))

	print("Picking minimum dataset size: ", set_size)
	#resize
	positive_set = positive_set[:set_size]
	negative_set = negative_set[:set_size]

	#concat
	export_set = positive_set + negative_set

	#shuffle the dataset
	print("Shuffle...")
	random.shuffle(export_set)
	random.shuffle(export_set)
	random.shuffle(export_set)
	random.shuffle(export_set)

	print("Spliting datasets into train and test: ")
	#split in training and testing sets
	half = len(export_set) >> 1	
	train_set = export_set[:half]
	test_set  = export_set[half:]

	print("Train set size: ", len(train_set))
	print("Test set size: ", len(test_set))

	with open(trainfile, 'w') as outfile:
		for tweet in train_set:
			json.dump(tweet, outfile)
			outfile.write('\n')
		outfile.close()
		print("Exporting to: ", trainfile)

	with open(testfile, 'w') as outfile:
		for tweet in test_set:
			json.dump(tweet, outfile)
			outfile.write('\n')
		outfile.close()
		print("Exporting to: ", testfile)


if __name__ == "__main__":
   main(sys.argv[1:])