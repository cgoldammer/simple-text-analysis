"""This runs a range of unit tests for simple-text-analysis
"""

import numpy as np
from pandas import DataFrame
import nltk
# This is required to make NLTK work with virtual environments. 
# Change the environment before using.
# nltk.data.path.append('/Users/cg/Dropbox/code/Python/nltk_data/')
from textblob import TextBlob
import unittest
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from pandas.util.testing import assert_frame_equal
from ..src import text_model as tm
from ..src.text_model import TextModel, modules_to_dictionary


texts = "one two three four five six seven eight nine ten eleven".split()
outcomes = range(len(texts))
texts_entities = ["I am Barack Obama", "I and Peter Mccain",
                  "I am James", "I loves Jim", "I love Hans",
                  "I am Jill", "I am Jack", "I am John",
                  "I am Jack", "I am Jack", "I am John"]


class TestFeaturizers(unittest.TestCase):
    def test_lemmatizer(self):
        """Testing that the lemmatizer works correctly"""
        lemmatizer = tm.Lemmatizer()
        unlemmatized = ["does", "goes", "seas"]
        lemmatized = lemmatizer.transform(unlemmatized)
        print "Lemmatized:"
        print lemmatized

        self.assertEquals(lemmatized, ["doe", "go", "sea"])

    def test_named_entity(self):
        text_train = "My name is John Adams and I am Scottie Pippen."
        text_test = "Michael Jordan is playing basketball with Scottie Pippen"

        named_entity_featurizer = tm.NamedEntityFeaturizer()
        named_entity_featurizer.fit([text_train], y=None)

        X_train = named_entity_featurizer.transform([text_train])
        X_test = named_entity_featurizer.transform([text_test])

        # In the training sample, both features are found.
        self.assertEqual(X_train.todense().tolist(), [[1, 1]],
                         "Training data not correctly featurized")
        # In the test sample, only the second feature (as ordered by
        # the training sample) is found
        self.assertEqual(X_test.todense().tolist(), [[0, 1]],
                         "Test data not correctly featurized")

    def test_aggregate(self):
        modules = ["aggregate"]
        text_model = TextModel(outcomes, texts, modules)
        summ = text_model.summary
        print summ.features


class TestTextModel(unittest.TestCase):
    def test_direction(self):
        """Testing that the prediction increases with words at that have 
        higher outcomes in the training data"""
        modules = ["bag-of-words"]
        options = {'remove-stopwords': False}
        text_model = TextModel(outcomes, texts, modules, options)
        predict_low = text_model.predict("one two three")
        predict_high = text_model.predict("nine ten eleven")
        self.assertTrue(predict_low < predict_high)

    def test_feature_union(self):
        """Tests that combining multiple featurizers works as expected"""
        modules = ["bag-of-words", "entities"]
        modules_list, _ = modules_to_dictionary(modules)
        feature_union = FeatureUnion(modules_list)
        feature_union.fit(texts_entities, outcomes)
        feature_union.transform(["unknown"])

    def test_modules_to_dictionary(self):
        """Testing that the functions correctly translates a variety of types 
        into a list and dictionary"""

        module_bag_of_words = ("bag-of-words", CountVectorizer())
        module_aggregate = ("aggregate", tm.AggregateFeaturizer())

        modules = ["bag-of-words", "aggregate"]
        modules_list, _ = tm.modules_to_dictionary(modules)
        expected_module_list = [module_bag_of_words, module_aggregate]
        for i in range(len(modules)):
            self.assertEqual(modules_list[i][0], expected_module_list[i][0])
            self.assertEqual(type(modules_list[i][1]), type(expected_module_list[i][1]))

    def test_create_basic_model(self):
        """Testing example usages"""

        options = {'remove-stopwords': False}

        modules = "bag-of-words"
        text_model = TextModel(outcomes, texts, modules, options)

        modules = ["bag-of-words"]
        text_model = TextModel(outcomes, texts, modules, options)

        modules = ["bag-of-words", "entities"]
        text_model = TextModel(outcomes, texts_entities, modules)

        modules = {"bag-of-words": {"max__features": None},
                   "emotions": True, "entities": True}

        # As of now, the named entities don't work with lower case. Needs
        # to be fixed.
        modules = ["bag-of-words", "aggregate"]
        options = {"lowercase": True, "lemmatize": True}
        text_model = TextModel(outcomes, texts_entities,
                               modules, options)



    def test_options(self):
        modules = "bag-of-words"
        options = {"lowercase": False}
        modules_list, options = modules_to_dictionary(modules)

        text_model = TextModel(outcomes, texts_entities,
                               modules, options)
        prediction_1 = text_model.predict(["James"])
        prediction_2 = text_model.predict(["james"])
        self.assertTrue(prediction_1 != prediction_2)

        options = {"lowercase": True}
        text_model_low = TextModel(outcomes, texts_entities,
                                   modules, options)
        prediction_1 = text_model_low.predict(["Barack"])
        prediction_2 = text_model_low.predict(["barack"])
        self.assertTrue(prediction_1 == prediction_2)

        options = {"lemmatize": False}
        text_model = TextModel(outcomes, texts_entities,
                               modules, options)
        prediction_1 = text_model.predict(["loves"])
        prediction_2 = text_model.predict(["love"])
        self.assertTrue(prediction_1 != prediction_2)

        options = {"lemmatize": True}
        text_model = TextModel(outcomes, texts_entities,
                               modules, options)
        prediction_low = text_model.predict(["loves"])
        prediction_high = text_model.predict(["loves"])
        self.assertTrue(prediction_low == prediction_high)

    def test_stopwords(self):
        modules = "bag-of-words"
        options = {"remove-stopwords": True}
        text_model = TextModel(outcomes, texts_entities,
                               modules, options)
        prediction_0 = text_model.predict(["and"])
        prediction_1 = text_model.predict([""])
        self.assertTrue(prediction_0 == prediction_1)

        options = {"remove-stopwords": False}
        text_model = TextModel(outcomes, texts_entities,
                               modules, options)
        prediction_0 = text_model.predict(["and"])
        prediction_1 = text_model.predict([""])
        self.assertTrue(prediction_0 != prediction_1)


class TestTextSummary(unittest.TestCase):
    modules = "bag-of-words"
    options = {"remove-stopwords": False}

    def setUp(self):
        """Creating a `TextModel` and extracting its summary"""
        self.text_model = TextModel(outcomes, texts, self.modules, self.options)
        self.summ = self.text_model.summary

    def test_std(self):
        """The matrix of standard deviations corresponds to the texts"""
        std_X = self.summ.std_X
        self.assertAlmostEqual(std_X.std(), 0)
        self.assertTrue(len(std_X) == len(texts))

    def test_regression_table(self):
        """The features `one` and `eleven` are most important"""
        table = self.summ.get_regression_table()
        print table
        first_two = table.index[0:2]
        self.assertTrue("bag-of-words__eleven" in first_two)
        self.assertTrue("bag-of-words__one" in first_two)

    def test_features(self):
        modules = ["bag-of-words", "aggregate"]
        print "Mods: %s" %modules
        text_model = TextModel(outcomes, texts_entities, modules, self.options)
        summ = text_model.summary
        modules_table = list(set(summ.get_regression_table().transformer))
        self.assertEquals(modules_table, modules)


if __name__ == '__main__':
    unittest.main()
