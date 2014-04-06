'''
Created on Apr 1, 2014

@author: Christian Goldammer
'''

import unittest
import text_model_functions

class TestTextModelFunctions(unittest.TestCase):
    
    minimal_csv="""outcome,text
1,"hello"
2,"yes" """

    outcomes=[1,2,3,4,5,1,2,3,4,5]
    texts=["hello","yes","no","why","is","hello","yes","no","why","is"]
    
    def setUp(self):
        pass
    
    def test_featurizer(self):
        modules=['CountVectorizer']
        featurizer=text_model_functions.FeaturizerMultipleModules(modules)
        # The number of options that needs to be set should be minimal
        #options={'max_features':10}
        featurizer.fit(self.texts,self.outcomes)
        X=featurizer.transform(self.texts)
        print "X matrix"
        print X
    
    def test_create_basic_model(self):
        
        modules=['CountVectorizer']
        text_model=text_model_functions.TextModel(self.outcomes,self.texts,modules,verbose=True)
        
        print text_model.predict(["Unknown"])
        
        # Testing the prediction with unknown word returns the mean
        #self.assertEqual(text_model.predict("Unknown"),1.5)
        
if __name__ == '__main__':
    unittest.main()  
    
