from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import gensim
import numpy as np
import random, re
MAX_NB_WORDS=20

def tweetsToVec(raw_tweets):
	exclude = '!"$%&\'()*+,-./:;<=>?¿[\\]^_`{|}~'
	regex = re.compile('[%s]' % re.escape(exclude))

	tweets = []

	tknzr = TweetTokenizer()
	stop_words = set(stopwords.words('spanish'))

	#check if it is a string or a list of strings
	if not isinstance(raw_tweets, list):
		raw_tweets = [raw_tweets]

	#Cleaning tweets
	for tweet in raw_tweets:
		#remove url
		no_url = re.sub(r"https?\S+", "", tweet)
		#no punctuation
		no_pun = regex.sub('', no_url)
		#tokenize
		tokenized = tknzr.tokenize(no_pun)
		#remove stop words
		important_words=[]
		for word in tokenized:
			if word not in stop_words and word[0] != '#':
				important_words.append(word)
				
		tweets.append(important_words)

	# load http://crscardellino.me/SBWCE/ trained model
	model = gensim.models.KeyedVectors.load_word2vec_format('data_process/SBW-vectors-300-min5.bin', binary=True)

	shape = (len(tweets), MAX_NB_WORDS, 300)
	tweets_tensor = np.zeros(shape, dtype=np.float32)

	for i in range(len(tweets)):
		#vectorizing each word in the tweet with a vector shape = (300,)
		length = len(tweets[i])
		for f in range(length):
			word = tweets[i][f]
			if f >= MAX_NB_WORDS:
				continue
			#if is not in the vocabulary
			if word in model.wv.vocab:
				tweets_tensor[i][f] = model.wv[word]
			else:
				#if it is a mention vectorize a name, for example @michael123 -> would be Carlos
				if word[0] == '@':
					tweets_tensor[i][f] = model.wv[name()]
				#if not append the unknown token
				else:
					tweets_tensor[i][f] = model.wv['unk']
		#End of sentence token
		if length - 1 < MAX_NB_WORDS:
			tweets_tensor[i][length - 1] = model.wv['eos']

	if len(tweets) == 1:
		return tweets_tensor
	else:
		return tweets_tensor
