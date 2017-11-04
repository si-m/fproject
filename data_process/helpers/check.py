#!/usr/bin/env python
import sys, getopt
import numpy as np
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments
import random

#Python program to check distribution of polarities in the dataset.
def main(argv):
	inputfile = ''
	positive_set, negative_set, neutral_set, export_set = [], [], [], []
	set_size = 0

	try:
		opts, args = getopt.getopt(argv,"hi:",["ifile="])
	except getopt.GetoptError:
		print('check.py -i <inputfile.json>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('check.py -i <inputfile.json>')
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
		if(raw['klass'] != 'NONE'):
			if raw['klass'] == "positive":
				positive_set.append({'text': raw['text'], 'klass': raw['klass']})
			if raw['klass'] == "negative":
				negative_set.append({'text': raw['text'], 'klass': raw['klass']})
			if raw['klass'] == "neutral":
				neutral_set.append({'text': raw['text'], 'klass': raw['klass']})

	print("Sizes: ")
	# check sizes
	print("Positive tweets:", len(positive_set))
	print("Negative tweets:", len(negative_set))
	print("Neutral tweets:", len(neutral_set))


	# smallest
	set_size = min(len(positive_set), len(negative_set), len(neutral_set))


if __name__ == "__main__":
   main(sys.argv[1:])