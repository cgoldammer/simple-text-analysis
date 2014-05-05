"""This runs a range of unit tests for simple-text-analysis
"""

import numpy as np
from pandas import DataFrame
import nltk
# This is required to make NLTK work with virtual environments. 
# Change the environment before using.
nltk.data.path.append('/Users/cg/Dropbox/code/Python/nltk_data/')
from textblob import TextBlob
import unittest
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from pandas.util.testing import assert_frame_equal
import text_model_functions
from text_model_functions import TextModel, modules_to_dictionary

class TestFeaturizers(unittest.TestCase):
    def test_lemmatizer(self):
        """Testing that the lemmatizer works correctly"""
        lemmatizer = text_model_functions.Lemmatizer()
        unlemmatized = ["does", "goes", "seas"]
        lemmatized = lemmatizer.transform(unlemmatized)
        print "Lemmatized:"
        print lemmatized

        self.assertEquals(lemmatized,["doe", "go", "sea"])

    def test_named_entity_featurizer(self):
        text_train = "My name is John Adams and I am Scottie Pippen."
        text_test = "Michael Jordan is playing basketball with Scottie Pippen"

        named_entity_featurizer = text_model_functions.NamedEntityFeaturizer()
        named_entity_featurizer.fit([text_train], y=None)

        X_train = named_entity_featurizer.transform([text_train])
        X_test = named_entity_featurizer.transform([text_test])

        # In the training sample, both features are found.
        self.assertEqual(X_train.tolist(), [[1,1]],
                           "Training data not correctly featurized")
        # In the test sample, only the second feature (as ordered by
        # the training sample) is found
        self.assertEqual(X_test.tolist(), [[0,1]],
                           "Test data not correctly featurized")

    def test_emotion_featurizer(self):
        """Testing that the emotion featurizer returns the same values 
        one would get from tagging sentiment by hand"""

        text = "I am happy and you better believe it."
        sentiment = TextBlob(text).sentiment

        emotion_featurizer = text_model_functions.EmotionFeaturizer()
        transformed = emotion_featurizer.fit_transform([text], y=None)

        print "Transformed: %s" % transformed

        self.assertEqual(transformed[0,0], sentiment.polarity)
        self.assertEqual(transformed[0,1], sentiment.subjectivity)


class TestTextModelFunctions(unittest.TestCase):

    texts_entities = ["I am Barack Obama", "I am John Mccain",
    "I am James", "I loves Jim", "I love John",
    "I am Jill", "I am Jack", "I am John",
    "I am Jack", "I am Jack", "I am John"]
    texts = "one two three four five six seven eight nine ten eleven".split()
    outcomes = range(len(texts_entities))

    def setUp(self):
        pass

    def test_direction(self):
        """Testing that the prediction increases with words at that have 
        higher outcomes in the training data"""
        modules = ["bag-of-words"]
        options = {'remove-stopwords': False}
        text_model = TextModel(self.outcomes, self.texts, modules, options)
        predict_low = text_model.predict("one two three")
        predict_high = text_model.predict("nine ten eleven")
        print "Prediction if group low: %s | high: %s" % (predict_low, predict_high)
        self.assertTrue(predict_low < predict_high)

    def test_feature_union(self):
        """Tests that combining multiple featurizers works as expected"""
        modules = ["bag-of-words", "emotions", "entities"]
        modules_list, _ = modules_to_dictionary(modules)
        feature_union = FeatureUnion(modules_list)
        feature_union.fit(self.texts_entities, self.outcomes)
        feature_union.transform(["unknown"])

    def test_modules_to_dictionary(self):
        """Testing that the functions correctly translates a variety of types 
        into a list and dictionary"""

        module_bag_of_words = ("bag-of-words", CountVectorizer())
        module_emotions = ("emotions", text_model_functions.EmotionFeaturizer())

        modules = ["bag-of-words", "emotions"]
        modules_list, _ = text_model_functions.modules_to_dictionary(modules)
        expected_module_list = [module_bag_of_words, module_emotions]
        for i in range(len(modules)):
            self.assertEqual(modules_list[i][0], expected_module_list[i][0])
            self.assertEqual(type(modules_list[i][1]), type(expected_module_list[i][1]))

        modules = {"bag-of-words": {"max__features": None},
                 'emotions':True, 'entities':True, 'topics':True
        }

    def test_create_basic_model(self):
        """Testing example usages"""
        
        options = {'remove-stopwords': False}
        
        modules="bag-of-words"
        text_model = TextModel(self.outcomes, self.texts, modules, options)

        modules = ["bag-of-words"]
        text_model = TextModel(self.outcomes, self.texts, modules, options)

        modules = ["bag-of-words", "emotions", "entities"]
        text_model = TextModel(self.outcomes, self.texts_entities, modules)

        modules = {"bag-of-words": {"max__features":None},
                 "emotions": True, "entities": True}

        # As of now, the named entities don't work with lower case. Needs
        # to be fixed.
        modules = ["bag-of-words", "emotions"]
        options = {"lowercase": True, "lemmatize": True}
        text_model = TextModel(self.outcomes, self.texts_entities,
                                 modules, options)
        print text_model.predict(["Unknown"])

    def test_options(self):
        modules = "bag-of-words"
        options = {"lowercase": False}
        modules_list, options = modules_to_dictionary(modules)
        
        text_model = TextModel(self.outcomes, self.texts_entities,
                                 modules, options)
        prediction_1 = text_model.predict(["James"])
        prediction_2 = text_model.predict(["james"])
        self.assertTrue(prediction_1 != prediction_2)

        options = {"lowercase": True}
        text_model_low = TextModel(self.outcomes, self.texts_entities,
                                 modules, options)
        prediction_1 = text_model_low.predict(["Barack"])
        prediction_2 = text_model_low.predict(["barack"])
        self.assertTrue(prediction_1 == prediction_2)

        options = {"lemmatize": False}
        text_model = TextModel(self.outcomes, self.texts_entities,
                                 modules, options)
        prediction_1 = text_model.predict(["loves"])
        prediction_2 = text_model.predict(["love"])
        self.assertTrue(prediction_1 != prediction_2)

        options = {"lemmatize": True}
        text_model = TextModel(self.outcomes, self.texts_entities,
                                 modules, options)
        prediction_low = text_model.predict(["loves"])
        prediction_high = text_model.predict(["loves"])
        self.assertTrue(prediction_low == prediction_high)

if __name__ == '__main__':
    unittest.main()
