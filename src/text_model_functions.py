"""This module contains all functions for simple-text-analysis. The package allows you
to build a predictive model from text in one line of code.

Example
-------
The following shows that it's easy to use the module::

    from text_model_functions import TextModel
    # A predictive text model requires outcomes and texts
    texts=["hello","yes","no","why","is","hello","yes","no","why","is","I am John Lennon"]
    outcomes=range(len(texts_entities))
    
    # Building a text model takes one line
    text_model=TextModel(outcomes,texts,'bag-of-words')
    # A text model allows you to predict the outcome for an arbitrary text
    text_model.predict("Jack Lennon")
    
"""



class RidgeWithStats(Ridge):
    """This class runs a ridge regression, but it adds an attribute for
the standard deviation of the feature matrix, which is useful
to analyze the prediction."""
    def fit(self,X,y,sample_weight=1.0):
        #self.std_X=X.std()
        return Ridge.fit(self,X,y,sample_weight)

class Lemmatizer:
    """This is a transformer for lemmatization that can be used in a pipeline"""
    wnl=WordNetLemmatizer()
    
    def fit(self,X,y,**fit_params):
        return self
    
    def fit_transform(self,X,y,**fit_params):
        print "Fit-transform for lemmatizer"
        print X
        
        transformed=self.fit(X,y,**fit_params).transform(X,**fit_params)
        
        print "Transformed:"
        print transformed
        
        return transformed
    
    def transform(self,X,**fit_params):
        X_lemmatized=[]
        print "Un-Lemmatized:"
        print X
        for text in X:
            textBlob=TextBlob(text)
            X_lemmatized.append(" ".join([self.wnl.lemmatize(word) for word in textBlob.words]))
            
        print "Lemmatized:"
        print X_lemmatized
        return X_lemmatized  
    
    def get_params(self,deep=False):
        return {} 

class NamedEntityFeaturizer:
    """ This is a transformer that turns text into the named entities contained."""
    types=['PERSON',"ORGANIZATION"]
    entities=[]
    entities_set=None
    
    def fit(self,X,y,**fit_params):
        text_all=" ".join(X)
        self.entities=entity_dict_to_list(named_entities(text_all,self.types))
        self.entities_set=set(self.entities)
        return self
    
    def fit_transform(self,X,y,**fit_params):
        self.fit(X,y,**fit_params).transform(X,**fit_params)
        return self.transform(X)
    
    def transform(self,X,**fit_params):
        X_data=[] 
        for text in X:
            entities_in_row=entity_dict_to_list(named_entities(text,self.types))
            X_data.append(position_list(self.entities_set,entities_in_row,verbose=False))
        X_data=np.array(X_data) 
        if X_data.shape[1]==0:
            raise ValueError("There are no named entitities in the training data!")
        return X_data
    
    def get_params(self,deep=False):
        return {}

def module_from_name(module):
    if module=="bag-of-words":
        return ("bag-of-words",CountVectorizer())
    if module=="emotions":
        return ("emotions",EmotionFeaturizer())
    if module=="entities":
        return ("entities",NamedEntityFeaturizer())
    
def modules_to_dictionary(modules):
    """The modules argument can be provided in a wide variety of types (string, list,dictionary).
    Internally, this will get translated to a list (of module names and modules) and
    a dictionary of options."""
    modules_list=[]
    options={}
    
    if type(modules)==str:
        modules_list.append(module_from_name(modules))
    if type(modules)==list:
        for module in modules:
            modules_list.append(module_from_name(module))
    if type(modules)==dict:
        for module in modules.keys():
            modules_list.append(module_from_name(module))
    
    return modules_list,options

def named_entities(text,types=["PERSON","ORGANIZATION"]):
    """This functions returns named entities from a text. The function adapted from emh's code (http://stackoverflow.com/users/2673189/emh)
    
    Parameters
    ----------
    text: str
        UTF-8 string
    types: list of strings
        Currently the list can include only "PERSON" and "ORGANIZATION"
    
    Returns
    -------
    dict
        Dictionary with one entry for each type of entity. For each of these entries, contains a list of strings with found entities
    """
    named_entities={'PERSON':[],'ORGANIZATION':[]}
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    for typ in types:
        for subtree in sentt.subtrees(filter=lambda t: t.node == typ):
            entity = ""
            for leaf in subtree.leaves():
                entity=entity+" "+leaf[0]
            named_entities[typ].append(entity.strip())
    return named_entities

def entity_dict_to_list(entity_dict):
    entities=[]
    for type in entity_dict.keys():
        entities.extend(["ENTITY__"+type+"_"+entity for entity in entity_dict[type]])
    return entities

def position_list(targets,sources,verbose=False):
    if verbose:
        print "Finding the positions of %s in %s" %(sources,targets)
    positions=[(target in sources) for target in targets]
    if verbose:
        print "Positions: %s" %positions
    positions=1*Series(positions)
    if verbose:
        print "Positions: %s" %positions
    return list(positions)

class EmotionFeaturizer:
    """This class is used to extract macro-features of the text. Currently, it includes
    sentiment and subjectivity"""
    types=["polarity","subjectivity"]
    
    def value_given_type(self,type,text):
        sentiment=TextBlob(text).sentiment
        if type=="polarity":
            return sentiment.polarity
        if type=="subjectivity":
            return sentiment.subjectivity
    
    def fit(self,X,y,**fit_params):
        return self
    
    def fit_transform(self,X,y,**fit_params):
        self.fit(X,y,**fit_params).transform(X,**fit_params)
        return self.transform(X)
    
    def transform(self,X,**fit_params):
        X_data=[]
        for text in X:
            text_data=[]
            for typ in self.types:
                text_data.append(self.value_given_type(typ,text))
            X_data.append(text_data)
        X_data=np.array(X_data) 
        return X_data
    
    def get_params(self,deep=False):
        return {}
    
class TextModel:
    """This is the main class from this module. It allows you to build a text model
    given outcomes, texts, text modules used, and options."""
    pipe = None
    regression_table=None
 
    def __init__(self,outcomes,texts,modules,options,verbose=False):
        
        data=DataFrame({"y":outcomes,"text":texts})       
        N=data.shape[0]
        data.index=[str(x) for x in range(N)]
       
        alphas=Series([.001,.01,.05,.1,.2,1,10])*N
        
        text_cleaner=TextCleaner(**options)
        
        modules_list,options_2=modules_to_dictionary(modules)
        if len(modules_list)==0:
            raise ValueError("No modules specified or found.")

        feature_union=FeatureUnion(modules_list)
        #feature_union=CountVectorizer()
        ridge_model=RidgeWithStats()
        
        pipeline_list=[('cleaner',text_cleaner)]
        if "lemmatize" in options and options["lemmatize"]:
            pipeline_list.append(('lemmatizer', Lemmatizer()))
        pipeline_list.append(('featurizer', feature_union))
        pipeline_list.append(('ridge_model', ridge_model))

        
        pipe = Pipeline(pipeline_list)
        
#         parameter_set_for_gridsearch = {'vectorizer__stop_words':("english",),
#                         'featurizer__ngram_range': ((1,1),),
#                         'featurizer__max_features': (5000,),
#                         'featurizer__binary':(False,), 
#                         'featurizer__lowercase':(True,),              
#                         'ridge_model__alpha': tuple(alphas),
#                         'ridge_model__normalize':(True,)} 
        
        parameter_set_for_gridsearch = {'ridge_model__alpha': tuple(alphas),
                        'ridge_model__normalize':(True,),
                        'featurizer__bag-of-words__lowercase':(False,), } 
        
        # Remove all parameters from the gridsearch that are set:
        #for key in parameters_initial.keys():
        #    if key in parameter_set_for_gridsearch:
        #        parameter_set_for_gridsearch.pop(key)
            
        #pipe.set_params(**parameters_initial)   
        
        # Pick meta-parameters using grid_search
        (parameters_gridsearch_result,pipe)=self.grid_search(data,pipe,parameter_set_for_gridsearch)
        if verbose:
            print "Results of grid-search:"
            print parameters_gridsearch_result
        
        parameters=parameters_gridsearch_result
        #for (key,value) in parameters_initial.iteritems():
        #    parameters[key]=value
        
        # I trust a regression only if the optimal regularization parameter alpha
        # is strictly inside the grid.
        data['y_hat']=pipe.predict(texts)    
        if verbose:
            #print "Parameters before grid-search:"
            #print parameters_initial
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
        if (len(texts)==1):
            return predictions[0]
        return predictions
    
    def get_features(self,text):
        x=Pipeline(self.pipe.steps[0:2]).transform([text]).toarray()[0]
        features=Series(self.features)[Series(x)>=1]
        tab=self.regression_table
        coefficients_for_features=tab[tab.index.isin(features)]
        return coefficients_for_features
   
class TextCleaner():
    """This function takes car of cleaning the text before it's featurized.
    
    Parameters
    ----------
    lowercase: bool
        Flag to convert the text to lowercase, defaults to false
    
    Returns
    -------
    class
        A transformer that can be used in a pipeline
    """
    
    lowercase=False
    
    options={}
    def __init__(self,**kwargs):
        self.options=kwargs
        if kwargs is not None:
            if "lowercase" in self.options:
                self.lowercase=self.options["lowercase"]
                
    def fit(self,X,y,**fit_params):
        return self
    
    def fit_transform(self,X,y,**fit_params):
        return self.fit(X,y,**fit_params).transform(X)
    
    def transform(self,X,**fit_params):
        rx = re.compile('[^a-zA-Z]+')
        X_cleaned=[rx.sub(' ', str(t)).strip() for t in X] 
        if self.lowercase:
            X_cleaned=[t.lower() for t in X_cleaned] 
        return X_cleaned
    
    def get_params(self,deep=False):
        # This needs to be fixed
        return self.options

