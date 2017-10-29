import sys, getopt
import gensim
import numpy as np
import json
from json_tricks.np import dump, dumps, load
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import random, re

tweets = []

tknzr = TweetTokenizer()
file = 'cleaned.json'

exclude = '!"#$%&\'()*+,-./:;<=>?¿[\\]^_`{|}~'
regex = re.compile('[%s]' % re.escape(exclude))
url=re.compile(r'\<http.+?\>', re.DOTALL)

stop_words = set(stopwords.words('spanish'))
with open('to_clean.json', 'r') as fpt:
	l = fpt.readlines()
for row, d in enumerate(l):
	try:
		pre_tweet = json.loads(str(d, encoding='utf-8'))
	except TypeError:
		pre_tweet = json.loads(d)

	#remove url
	no_url = re.sub(r"https?\S+", "", pre_tweet['text'])
	#no punctuation
	no_pun = regex.sub('', no_url)
	tokenized =tknzr.tokenize(no_pun)
	#remove stop words
	important_words=[]
	for word in tokenized:
		if word not in stop_words:
			important_words.append(word)

	tweets.append(important_words)

with open(file, 'w') as outfile:
	for tweet in tweets:
		json.dump(tweet, outfile)
		outfile.write('\n')
	outfile.close()
	print("Exporting to: ", file)



