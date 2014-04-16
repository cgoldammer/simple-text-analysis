import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import Ridge
import re
import math
import random
from operator import itemgetter
import nltk
# This is required to make NLTK work with virtual environments. Change the environment before using.
nltk.data.path.append('/Users/cg/Dropbox/code/Python/nltk_data/')
from nltk import word_tokenize, wordpunct_tokenize
import pickle
from sklearn.grid_search import GridSearchCV
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob
from scipy import sparse