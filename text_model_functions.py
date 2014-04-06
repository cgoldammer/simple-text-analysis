import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import Ridge
import re
import math
import random
from operator import itemgetter
import nltk
nltk.data.path.append('/Users/cg/Dropbox/code/Python/nltk_data/')
from nltk import word_tokenize, wordpunct_tokenize
import pickle
from sklearn.grid_search import GridSearchCV

# This class runs a ridge regression, but it adds an attribute for
# the standard deviation of the feature matrix, which is useful
# to analyze the prediction.
class RidgeWithStats(Ridge):
    def fit(self,X,y,sample_weight=1.0):
        self.std_X=X.std()
        return Ridge.fit(self,X,y,sample_weight)

# This class combines featurizers - this is in contrast to a pipe, which chains
# functions.
class FeaturizerMultipleModules:
    #options={'ngram_range':(1,1),'max_features':2,'binary':True,'lowercase':True}
    modules=None
    featurizers={}
    feature_names=[]
    def __init__(self,modules=['CountVectorizer']):
        self.modules=modules
        for module in modules:
            if module=="CountVectorizer":
                featurizer=CountVectorizer(max_features=10)
                self.featurizers[module]=featurizer
                
    def fit(self,X,y,**fit_params):
        self.feature_names=[]
        for module in self.modules:
            if module=="CountVectorizer":
                featurizer=self.featurizers[module]
                featurizer.fit(X,y,**fit_params)
                self.feature_names.append([module+"__"+feature for feature in featurizer.get_feature_names()])
    
    def fit_transform(self,X,y,**fit_params):
        self.fit(X,y,**fit_params)
        return self.transform(X)
    
    def transform(self,X,**fit_params):
        X_matrices={}
        X_matrix_full=None
        # Concatenate these matrices along their columns
        for module in self.modules:
            featurizer=self.featurizers[module]
            X_matrix=DataFrame(featurizer.transform(X,**fit_params).todense())
            if not X_matrix_full:
                X_matrix_full=X_matrix
            else:
                X_matrix_full=X_matrix_full.append(X_matrix,dim=0)
            X_matrix_full.columns=self.feature_names
        return X_matrix_full

    def get_params(self,deep=False):
        return {}
    
class TextModel:
    regression_table=None
 
    def __init__(self,outcomes,texts,modules,parameters_initial={},verbose=False):
        
        data=DataFrame({"y":outcomes,"text":texts})       
        N=data.shape[0]
        data.index=[str(x) for x in range(N)]
        
        if verbose:
            print "Data:"
            print data[0:10]
       
        alphas=Series([.001,.01,.05,.1,.2,1,10])*N
      
        pipe = Pipeline([
                         ('cleaner',TextCleaner()),
                         ('featurizer', FeaturizerMultipleModules(modules=modules)),
                         ('ridge_model', RidgeWithStats())])
        
        pipe = Pipeline([
                         ('cleaner',TextCleaner()),
                         ('featurizer', FeaturizerMultipleModules(modules=modules)),
                         ('ridge_model', RidgeWithStats())])
        
#         parameter_set_for_gridsearch = {'vectorizer__stop_words':("english",),
#                         'featurizer__ngram_range': ((1,1),),
#                         'featurizer__max_features': (5000,),
#                         'featurizer__binary':(False,), 
#                         'featurizer__lowercase':(True,),              
#                         'ridge_model__alpha': tuple(alphas),
#                         'ridge_model__normalize':(True,)} 
        
        parameter_set_for_gridsearch = {'ridge_model__alpha': tuple(alphas),
                        'ridge_model__normalize':(True,)} 
        
        # Remove all parameters from the gridsearch that are set:
        for key in parameters_initial.keys():
            if key in parameter_set_for_gridsearch:
                parameter_set_for_gridsearch.pop(key)
            
        pipe.set_params(**parameters_initial)   
        
        # Pick meta-parameters using grid_search
        (parameters_gridsearch_result,pipe)=self.grid_search(data,pipe,parameter_set_for_gridsearch)
        if verbose:
            print "Results of grid-search:"
            print parameters_gridsearch_result
        
        parameters=parameters_gridsearch_result
        for (key,value) in parameters_initial.iteritems():
            parameters[key]=value
        
        # I trust a regression only if the optimal regularization parameter alpha
        # is strictly inside the grid.
        data['y_hat']=pipe.predict(texts)    
        if verbose:
            print "Parameters before grid-search:"
            print parameters_initial
            print "Parameters from grid-search:"
            print parameters_gridsearch_result
            print "All parameters:"
            print parameters
        
        #if verbose:
        #print "Alpha from grid search: %s" %parameters['ridge_model__alpha']
        if parameters['ridge_model__alpha']==max(alphas):
            raise(ValueError("Regularization parameter hitting upper bound"))
        
        #self.coef_normalized=pipe.steps[-1][-1].coef_
        ridge=pipe.named_steps['ridge_model']
        self.coef=ridge.coef_
#         self.std_X=ridge.std_X
        self.pipe=pipe
#         self.parameters=parameters 

        #self.features=self.pipe.named_steps['vectorizer'].get_feature_names()  
        #self.regression_table=self.get_regression_table() 

    pipe = None
   
    def grid_search(self, train,pipe, params):
        """
        Train a model, optimizing over meta-parameters using sklearn's grid_search
        Returns the best estimator from a sklearn.grid_search operation
        (See http://scikit-learn.org/stable/modules/grid_search.html)        
        """
        grid_search = GridSearchCV(estimator=pipe, 
                                   param_grid=params,
                                   verbose=1)
        grid_search.fit(train.text,train.y)
        
        return grid_search.best_params_,grid_search.best_estimator_
        
    def save_pipe(self, fp):
        pickle.dump(self, fp)
        
    def load_pipe(self, fp):
        m = pickle.load(fp)
        self.pipe=m.pipe 
    
    def predict(self,texts):
        texts_original=texts
        # Put text into pipeline
        if isinstance(texts,(str, unicode)):
            texts=[texts]
        predictions=self.pipe.predict(texts)
        return predictions
    
    def get_features(self,text):
        """This takes a text and translates it into features. This function thus describes central logic of the TextModel"""
        x=Pipeline(self.pipe.steps[0:2]).transform([text]).toarray()[0]
        features=Series(self.features)[Series(x)>=1]
        tab=self.regression_table
        coefficients_for_features=tab[tab.index.isin(features)]
        return coefficients_for_features
   
class TextCleaner():
    
    def __init__(self,**other):
        pass
    
    def fit(self,X,y,**fit_params):
        pass
    
    def fit_transform(self,X,y,**fit_params):
        return self.transform(X)
    
    def transform(self,X,**fit_params):
        rx = re.compile('[^a-zA-Z]+')
        X_cleaned=[rx.sub(' ', str(t)).strip().lower() for t in X] 
        return X_cleaned
    
    def get_params(self,deep=False):
        return {"a":1}

