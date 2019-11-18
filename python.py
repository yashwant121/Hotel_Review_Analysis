import numpy as np 
import pandas as pd 
from nltk import word_tokenize
import nltk
nltk.download('sentiwordnet')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string 
from collections import Counter
from collections import defaultdict
from nltk.corpus import wordnet

from nltk.corpus import sentiwordnet
import spacy 
from prettytable import PrettyTable
nlp = spacy.load('en_core_web_sm')
letz = WordNetLemmatizer()


reviews = pd.read_csv('deceptive-opinion.csv')

x = reviews['text']

def sentenceTokenizer(para):
	tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
	return tokenizer.tokenize(x)

def posTag(text):
	tagged_text_list = []
	tagged_text_list.append(nltk.pos_tag(word_tokenize(text)))
	return tagged_text_list

def featureTags(pos_tagged_list):
 
    for i in pos_tagged_list:
        for j in i:
            if(j[1]=='NNS' or j[1]=='NNP' or j[1]=='NNPS' or j[1]=='NN' ):
	            return j[0]

	#return preWord












def mostCommonFeature(feature):
	col_count = Counter(feature)
	newList = []
	#print(col_count)
	for k,v in col_count.items():
		if(v >= 5 and k is not None):
			newList.append(k)
    
	print(newList)
	return newList

def orientation(inputWord):
	wordSynset = wordnet.synsets(inputWord)
	if((len(wordSynset)) != 0):
		word = wordSynset[0].name()
		orientation = sentiwordnet.senti_synset(word)
		if(orientation.pos_score() > orientation.neg_score()):
			return True
		elif(orientation.pos_score() < orientation.neg_score()):
			return False
		

def pos_words(sentence, token, ptag):
	sentence = [sent for sent in sentence.sents if token in sent.string]
	pwrds = []
	for sent in sentence:
		for word in sent:
			if token in word.string:
				pwrds.extend([child.string.strip() for child in word.children
													if child.pos_ == ptag])
	return pwrds

def opinionIdentification(tokenized_sentence, most_common_features):
	negwordList = {"dont't","never","nothing","nowhere","noone","none","not",
					"hasn't","hadn't","can't","couldn't","shouldn't","won't",
					"wouldn't","don't","doesn't","didn't","isn't","aren't","ain't"}

	opinionList = {}
	orientationList = {}
	t = PrettyTable(['Feature','Positive', 'Negative'])
	posAvg = 0.0
	negAvg = 0.0
	cAvg = 0
	for feature in most_common_features:
		feature = feature.lower()
		count = 0
		opinionList.setdefault(feature,[0,0,0])
		for sentence in tokenized_sentence:
			neg = False
			sentence = sentence.lower()
			if feature in sentence:
				#sentence = unicode(sentence)
				sentence = nlp(sentence)
				pwrds = pos_words(sentence, feature, 'ADJ')
				for word in pwrds:
					count += 1
					if word in orientationList:
						wordOrien = orientationList[word]
					else:
						wordOrien = orientation(word)
						orientationList[word] = wordOrien
					if word in negwordList:
						neg = True
					if neg is True and wordOrien is not None:
						wordOrien = not wordOrien
					if wordOrien is True:
						opinionList[feature][0] += 1
					elif wordOrien is False:
						opinionList[feature][1] += 1
					elif wordOrien is None:
						count -= 1
		if count > 0:
			opinionList[feature][0] = round(100*opinionList[feature][0]/count,2)
			opinionList[feature][1] = round(100*opinionList[feature][1]/count,2)

		if opinionList[feature][0] != 0 and opinionList[feature][1] != 0:
			t.add_row([feature,opinionList[feature][0],opinionList[feature][1]])
			posAvg -= opinionList[feature][0]
			negAvg -= opinionList[feature][1]
			cAvg += 1
	print (t)
	
	#print ("overall: ", posAvg/cAvg, "Positive ",negAvg/cAvg,"Negative")

X = reviews[(reviews['hotel'] == 'hyatt')]
print (len(X))
X = X['text']


feature = []
tokenized_sentence = []
for x in X:
	for sentence in sentenceTokenizer(x):
		tokenized_sentence.append(sentence)


for review in tokenized_sentence:
	pos_tagged_list = posTag(review)
	feature.append(featureTags(pos_tagged_list))


most_common_features = mostCommonFeature(feature)


opinionIdentification(tokenized_sentence, most_common_features)