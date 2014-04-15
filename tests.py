'''
Created on Apr 1, 2014

@author: Christian Goldammer
'''

import unittest
import text_model_functions
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pandas import DataFrame,Series
from pandas.util.testing import assert_frame_equal
from sklearn.pipeline import Pipeline,FeatureUnion

class TestFeaturizers(unittest.TestCase):
    def test_lemmatizer(self):
        """Testing that the lemmatizer works correctly"""
        lemmatizer=text_model_functions.Lemmatizer()
        unlemmatized=[["does","goes","seas"]]
        lemmatized=lemmatizer.transform(unlemmatized)
        print "Lemmatized:"
        print lemmatized
        
        self.assertEquals(lemmatized,[["doe","go","sea"]])
            
    def test_named_entity_featurizer(self):
        text_train="My name is John Adams and I am  as important as Scottie Pippen."
        text_test="Michael Jordan is playing basketball with Scottie Pippen"
                
        named_entity_featurizer=text_model_functions.NamedEntityFeaturizer()
        named_entity_featurizer.fit([text_train],y=None)
        
        X_train=named_entity_featurizer.transform([text_train])
        X_test=named_entity_featurizer.transform([text_test])
        
        print X_train
        print X_test
        
        # In the training sample, both features are found.
        assert_frame_equal(X_train, DataFrame([[1,1]]), "Training data not correctly featurized")
        # In the test sample, only the second feature (as ordered by
        # the training sample) is found
        assert_frame_equal(X_test, DataFrame([[0,1]]), "Test data not correctly featurized")
        
    def test_emotion_featurizer(self):
        """Testing that the emotion featurizer returns the same values one would get from
    tagging sentiment by hand"""
        
        text="I am happy and you better believe it."
        sentiment=TextBlob(text).sentiment
        
        emotion_featurizer=text_model_functions.EmotionFeaturizer()
        transformed=emotion_featurizer.fit_transform([text],y=None)
        types=emotion_featurizer.types
        
        print "Transformed: %s" %transformed
        
        self.assertEqual(transformed.ix[0,0], sentiment.polarity)
        self.assertEqual(transformed.ix[0,1], sentiment.subjectivity)
        
class TestTextModelFunctions(unittest.TestCase):
    
    minimal_csv="""outcome,text
1,"hello"
2,"yes" """

    
    texts=["hello","yes","no","why","is","hello","yes","no","why","is","I am Barack Obama"]
    texts_entities=["I am Barack Obama","I am John Mccain",
    "I am James","I loves Jim","I love John",
    "I am Jill","I am Jack","I am John",
    "I am Jack","I am Jack","I am John"]
    outcomes=range(len(texts_entities))
    
    def setUp(self):
        pass
    
    def test_feature_union(self):
        """Tests that combining multiple featurizers works as expected"""
        modules=['bag-of-words','emotions','entities']
        modules_list,options=text_model_functions.modules_to_dictionary(modules)
        feature_union=FeatureUnion(modules_list)
        feature_union.fit(self.texts,self.outcomes)
        feature_union.transform(["unknown"])
        
        
    def test_modules_to_dictionary(self):
        """Testing that the functions correctly translates a variety of types into a list and dictionary"""
        
        module_bag_of_words=('bag-of-words',CountVectorizer())
        module_emotions=('emotions',text_model_functions.EmotionFeaturizer())

        modules=['bag-of-words','emotions']
        modules_list,options=text_model_functions.modules_to_dictionary(modules)
        expected_module_list=[module_bag_of_words,module_emotions]
        for i in range(len(modules)):
            self.assertEqual(modules_list[i][0],expected_module_list[i][0])
            self.assertEqual(type(modules_list[i][1]),type(expected_module_list[i][1]))

        modules={'bag-of-words':{'max__features':None},
                 'emotions':True,
                 'entities':True,
                 'topics':True
        }
        
    def test_create_basic_model(self):
        
        # The following are example usages
        modules='bag-of-words'
        text_model=text_model_functions.TextModel(self.outcomes,self.texts,modules,options=None,verbose=True)
        
        modules=['bag-of-words']
        text_model=text_model_functions.TextModel(self.outcomes,self.texts,modules,options=None,verbose=True)
        
        modules=['bag-of-words','emotions','entities']
        text_model=text_model_functions.TextModel(self.outcomes,self.texts_entities,modules,options=None,verbose=True)
        
        modules={'bag-of-words':{'max__features':None},
                 'emotions':True,
                 'entities':True,
        }
        
        options={'lowercase':True,
                 'lemmatize':True,
        }
        text_model=text_model_functions.TextModel(self.outcomes,self.texts_entities,modules,options,verbose=True)
        print text_model.predict(["Unknown"])
        
    def test_options(self):
        modules='bag-of-words'
        options={'lowercase':False}
        modules_list,options=text_model_functions.modules_to_dictionary(modules)
        text_model=text_model_functions.TextModel(self.outcomes,self.texts_entities,modules,options,verbose=True)
        self.assertTrue(text_model.predict(["James"])!=text_model.predict(["james"]))
        
        options={'lowercase':True}
        text_model_low=text_model_functions.TextModel(self.outcomes,self.texts_entities,modules,options,verbose=True)
        self.assertTrue(text_model_low.predict(["Barack"])==text_model_low.predict(["barack"]))
        
        options={'lemmatize':False}
        modules_list,options=text_model_functions.modules_to_dictionary(modules)
        text_model=text_model_functions.TextModel(self.outcomes,self.texts_entities,modules,options,verbose=True)
        self.assertTrue(text_model.predict(["loves"])!=text_model.predict(["love"]))
        
        options={'lemmatize':True}
        text_model_low=text_model_functions.TextModel(self.outcomes,self.texts_entities,modules,options,verbose=True)
        self.assertTrue(text_model_low.predict(["loves"])==text_model_low.predict(["love"]))
        
        
if __name__ == '__main__':
    unittest.main()  
    
