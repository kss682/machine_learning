import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import movie_reviews
import random


documents=[(list(movie_reviews.words(fileid)),category) 
		for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]


print(documents[0][1])

random.shuffle(documents)

all_words=movie_reviews.words()
all_words=nltk.FreqDist(all_words)
all_words=list(all_words.keys())
print(all_words)


