#!/usr/bin/env python
import sys, getopt
import numpy as np
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments
import random

def main(argv):
	negative_file = 'spanish_tweets_neg_clean.txt'
	neg_outfile = 'neg_set.json'
	set_size = 0
	neg_set = []

	with open(negative_file, 'r') as fpt:
		l = fpt.readlines()
	for row, d in enumerate(l):
		neg_set.append({"text": d.rstrip(), "klass": "negative", "id": row})

	with open(neg_outfile, 'w') as outfile:
		for tweet in neg_set:
			json.dump(tweet, outfile)
			outfile.write('\n')
		outfile.close()

if __name__ == "__main__":
   main(sys.argv[1:])