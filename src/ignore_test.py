'''
Created on Apr 8, 2014

@author: cg
'''
import nltk
nltk.data.path.append('/Users/cg/Dropbox/code/Python/nltk_data/')






measurements = [
                    {'city': 'Dubai', 'temperature': 33.},
                    {'city': 'London', 'temperature': 12.},
                    {'city': 'San Fransisco', 'temperature': 18.},
                    ]

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vec = DictVectorizer()

vec.fit_transform(measurements).toarray()

vec.get_feature_names()

vectorizer = CountVectorizer(min_df=1)
corpus = [
          'This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?',
          ]
X = vectorizer.fit_transform(corpus)
X 

analyze = vectorizer.build_analyzer()
analyze("This is a text document to analyze.") 

vectorizer.transform(['Something completely new.']).toarray()

ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5,5), min_df=1)
counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
ngram_vectorizer.get_feature_names()

from nltk import word_tokenize   
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl=WordNetLemmatizer()
    def __call__(self,doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vect = CountVectorizer(tokenizer=LemmaTokenizer())  
corpus = [
          'This is the first document.',
          'This is the second second document.',
          'And the ! third one.',
          'Is this the first document?',
          ]
X = vect.fit_transform(corpus)
X 
vect.get_feature_names()
vect.transform(['Something! completely new.']).toarray()