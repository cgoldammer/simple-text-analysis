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

    outcomes=[1,2,3,4]
    texts=["hello","yes","no","why"]
    
    def setUp(self):
        pass
    
    def test_featurizer(self):
        modules=['CountVectorizer']
        featurizer=text_model_functions.FeaturizerMultipleModules(modules)
        # The number of options that needs to be set should be minimal
        options={'max_features':N}
        X=featurizer.transform(self.texts,**options)
    
    def test_create_basic_model(self):
        
        modules=['CountVectorizer']
        text_model=text_model_functions.TextModel(outcomes,texts,modules)
        
        print text_model.predict("Unknown")
        
        # Testing the prediction with unknown word returns the mean
        self.assertEqual(text_model.predict("Unknown"),1.5)
        
if __name__ == '__main__':
    unittest.main()  
    
