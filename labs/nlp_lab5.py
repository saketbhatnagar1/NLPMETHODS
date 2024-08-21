from nltk.util import ngrams
from collections import Counter

#Training Data:

corpus = [
    'This is a dog',
    'This is a cat',
    'I love my cat',
    'This is my name'
]

def preprocess(text):
  return text.lower().split()

all_words = []
for sentence in corpus:
  all_words.extend(preprocess(sentence))

#generate unigrams

unigrams = list(ngrams(all_words, 1))
bigrams = list(ngrams(all_words, 2))
trigrams = list(ngrams(all_words, 3))

print("Unigrams :")
unigram_count = Counter(unigrams)

for unigram, count in unigram_count.items():
    print(f"{unigrams}: {count}")


print("Unigrams :")
trigram_count = Counter(trigrams)

for trigram, count in trigram_count.items():
    print(f"{bigrams}: {count}")

import nltk
from nltk.corpus import gutenberg,brown
from nltk.util import ngrams
from collections import Counter

nltk.download('gutenberg')
nltk.download('brown')

corpus = gutenberg.words('austen-emma.txt') 


import nltk
from nltk.corpus import gutenberg,brown
from nltk.util import ngrams
from collections import Counter

nltk.download('gutenberg')
nltk.download('brown')

corpus = gutenberg.words('austen-emma.txt')
