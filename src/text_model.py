"""This module contains all functions for simple-text-analysis. The package
allows you to build a predictive model from text in one line of code. 
This package takes care of a lot of non-trivial choices (such as text 
cleaning, estimation, and validation, via
sensible defaults.

Example
-------
The following shows that it's easy to use the module::

    from text_model_functions import TextModel
    # A predictive text model requires outcomes and texts
    texts=["hello", "yes", "no", "why", "is", "hello",
           "yes", "no", "why", "is", "I am John Lennon"]
    outcomes=range(len(texts_entities))

    # Building a text model takes one line
    text_model=TextModel(outcomes,texts,'bag-of-words')
    # A text model allows you to predict the outcome for an arbitrary text
    text_model.predict("Jack Lennon")

"""

import numpy as np
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
import re
import nltk
# This is required to make NLTK work with virtual environments.
# Change the environment before using.
# nltk.data.path.append("/Users/cg/Dropbox/code/Python/nltk_data/")
import pickle
from sklearn.grid_search import GridSearchCV
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import collections
import pandas as pd
import itertools
from scipy.sparse import csr_matrix

nltk.data.path.append('/nltk_data/')


class RidgeWithStats(Ridge):
    def fit(self, X, y, sample_weight=1.0):
        self.std_X = DataFrame(X.toarray()).std()
        return Ridge.fit(self, X, y, sample_weight)


# Define a global dictionary with class subjects.
def module_from_name(module):
    if module == "bag-of-words":
        return ("bag-of-words", CountVectorizer())
    if module == "emotions":
        return ("emotions", EmotionFeaturizer())
    if module == "entities":
        return ("entities", NamedEntityFeaturizer())
    if module == "aggregate":
        return ("aggregate", AggregateFeaturizer())
    else:
        raise ValueError("Unknown module!")


def modules_to_dictionary(modules):
    """The modules argument can be provided in a wide variety of types 
    (string, list,dictionary). Internally, this will get translated to a 
    list (of module names and modules) and a dictionary of options."""
    modules_list = []
    options = {}
    # isinstance. And transform into dictionary. Also use 'for' regardless
    # of type
    if type(modules) == str:
        modules_list.append(module_from_name(modules))
    if type(modules) == list:
        for module in modules:
            modules_list.append(module_from_name(module))
    if type(modules) == dict:
        for module in modules.keys():
            modules_list.append(module_from_name(module))

    return modules_list, options

def named_entities(text, types=None):
    """This functions returns named entities from a text.
    Adapted from emh's code (http://stackoverflow.com/users/2673189/emh)

    Parameters
    ----------
    text: str
        UTF-8 string
    types: list of strings
        Currently the list can include only "PERSON" and "ORGANIZATION"

    Returns
    -------
    dict
        Dictionary with one entry for each type of entity. For each of these 
        entries, contains a list of strings with found entities
    """
    if not types:
        types = ["PERSON", "ORGANIZATION"]
    named_entities = {"PERSON": [], "ORGANIZATION": []}
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary=False)
    for type_ in types:
        for subtree in sentt.subtrees(filter=lambda t: t.label() == type_):
            entity = ""
            for leaf in subtree.leaves():
                entity = entity + " " + leaf[0]
            named_entities[type_].append(entity.strip())
    return named_entities


def entity_dict_to_list(entity_dict):
    entities = []
    # Note: Use iterators.
    for type_ in entity_dict.keys():
        ent_type = entity_dict[type_]
        entities.extend(["ENTITY__" + type_ + "_" + e for e in ent_type])
    return entities


def position_list(targets, sources):
    positions = [(target in sources) for target in targets]
    positions = 1 * Series(positions)
    return list(positions)


class BaseTransformer:
    def fit_transform(self, X, y, **fit_params):
        transformed = self.fit(X, y, **fit_params).transform(X, **fit_params)
        return transformed

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, X, **fit_params):
        pass

    def get_params(self, deep=False):
        return {}


# Start from base class and then override function transform_word
class Lemmatizer(BaseTransformer):
    """This is a transformer for lemmatization."""
    wnl = WordNetLemmatizer()

    def transform(self, X, **fit_params):
        X_lemmatized = [" ".join(
            [self.wnl.lemmatize(word) for word in TextBlob(text).words])
                        for text in X]
        return X_lemmatized

    def get_params(self, deep=False):
        return {}


class NamedEntityFeaturizer(BaseTransformer):
    """This is a transformer that turns text into named entities."""
    types = ["PERSON", "ORGANIZATION"]
    entities = []
    entities_set = None

    def fit(self, X, y, **fit_params):
        text_all = " ".join(X)
        entities = named_entities(text_all, self.types)
        self.entities = entity_dict_to_list(entities)
        self.entities_set = set(self.entities)
        return self

    def transform(self, X, **fit_params):
        X_data = []
        for text in X:
            entities = named_entities(text, self.types)
            entities_in_row = entity_dict_to_list(entities)
            positions = position_list(self.entities_set, entities_in_row)
            X_data.append(positions)
        X_data = np.array(X_data)
        if X_data.shape[1] == 0:
            raise ValueError("No named entities in training data!")
        return csr_matrix(X_data)

    def get_feature_names(self):
        return self.entities_set


class EmotionFeaturizer(BaseTransformer):
    """This class is used to extract macro-features of the text.
    Currently, it includes sentiment and subjectivity"""
    types = ["polarity", "subjectivity"]

    def value_given_type(self, type_, text):
        sentiment = TextBlob(text).sentiment._asdict()
        return sentiment[type_]

    def transform(self, X, **fit_params):
        X_data = []
        for text in X:
            text_data = []
            # Use list comprehension
            for type_ in self.types:
                text_data.append(self.value_given_type(type_, text))
            X_data.append(text_data)
        X_data = np.array(X_data)
        return csr_matrix(X_data)

    def get_feature_names(self):
        return TextBlob("").sentiment._asdict().keys()


def aggregate_features(text):
    f = collections.OrderedDict()
    f['length'] = len(text)
    f['questions'] = text.count('?')
    f['exclamation'] = text.count('!')

    sentences = TextBlob(text).sentences
    f['number of sentences'] = len(sentences)
    lengths = Series([len(s) for s in sentences])
    f['length_std'] = lengths.std() if f['number of sentences'] > 1 else 0

    return f



class AggregateFeaturizer(BaseTransformer):
    """Extracts aggregate features from the text"""
    def transform(self, X, **fit_params):
        X_data = np.array([aggregate_features(text).values() for text in X])
        return csr_matrix(X_data)

    def get_feature_names(self):
        return aggregate_features("").keys()

class TextModel:
    """This is the main class from this module. It allows you to build a
    text model given outcomes, texts, text modules used, and options."""
    pipe = None
    regression_table = None

    def __init__(self, outcomes, texts, modules, options=None):

        # Setting the default options
        if not options:
            options = {'lemmatize': False,
                       'lowercase': False,
                       'remove-stopwords': True}

        data = DataFrame({"y": outcomes, "text": texts})
        self.is_dummy_outcome = set(data.y) == set([0, 1])
        N = data.shape[0]
        self.number_of_observations = N
        data.index = [str(x) for x in range(N)]

        # Defining the alphas for the cross-validation. Note that
        # alpha scales proportionally with the number of observations.
        number_of_alphas = 5
        logspace_min = -2
        alphas = Series(np.logspace(logspace_min,
                                    logspace_min + number_of_alphas - 1,
                                    number_of_alphas)) * N

        text_cleaner = TextCleaner(**options)

        modules_list, _ = modules_to_dictionary(modules)
        if len(modules_list) == 0:
            raise ValueError("No modules specified or found.")

        feature_union = FeatureUnion(modules_list)
        ridge_model = RidgeWithStats()

        pipeline_list = [('cleaner', text_cleaner)]
        if options.get('lemmatize'):
            pipeline_list.append(('lemmatizer', Lemmatizer()))
        pipeline_list.append(('featurizer', feature_union))
        pipeline_list.append(('ridge_model', ridge_model))

        pipe = Pipeline(pipeline_list)

        parameters_initial = {'ridge_model__normalize': True}
        # If bag-of-words is included, add the relevant parameters
        if 'bag-of-words' in modules:
            def vec_name(param):
                return ('featurizer__bag-of-words__' + param)

            parameters_initial[vec_name('lowercase')] = False
            parameters_initial[vec_name('ngram_range')] = (1, 1)
            if options.get('remove-stopwords'):
                parameters_initial[vec_name('stop_words')] = "english"

        parameter_for_gridsearch = {'ridge_model__alpha': tuple(alphas)}

        pipe.set_params(**parameters_initial)

        # Pick meta-parameters using grid_search.
        grid_result = self.grid_search(data, pipe, parameter_for_gridsearch)
        (parameters_gridsearch_result, pipe) = grid_result
        # I trust a regression only if the optimal regularization
        # parameter alpha is strictly inside the grid.
        if parameters_gridsearch_result["ridge_model__alpha"] == max(alphas):
            error = "Regularization parameter hitting upper bound"
            error+= "Result: %s" %str(grid_result)
            error+= str(data)
            raise ValueError(error)

        # The full parameters consist of the initial values
        # and the parameters found through the grid-search
        parameters = parameters_initial
        for (key, value) in parameters_gridsearch_result.iteritems():
            parameters[key] = value

        # Keeping the regression, its coefficients, and the pipe as attributes
        ridge = pipe.named_steps["ridge_model"]
        self.coef = ridge.coef_
        self.pipe = pipe
        self.parameters = parameters
        self.modules = modules

        self.summary = ModelSummary(self, data)


    def grid_search(self, train, pipe, params):
        """
        Train a model, optimizing over meta-parameters using sklearn's
        grid_search. Returns the best estimator from a sklearn.grid_search
        operation
        (See http://scikit-learn.org/stable/modules/grid_search.html)
        
        Function contributed by Ryan Wang.
        """
        grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=2)
        grid_search.fit(train.text, train.y)
        return grid_search.best_params_, grid_search.best_estimator_

    def save_pipe(self, fp):
        pickle.dump(self, fp)

    def load_pipe(self, fp):
        m = pickle.load(fp)
        self.pipe = m.pipe

    def predict(self, texts):
        if isinstance(texts, basestring):
            texts = [texts]
        predictions = self.pipe.predict(texts)
        if (len(texts) == 1):
            return predictions[0]
        return predictions


class TextCleaner(BaseTransformer):
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

    lowercase = False
    options = {}

    def __init__(self, **kwargs):
        self.options = kwargs
        if kwargs is not None:
            if "lowercase" in self.options:
                self.lowercase = self.options["lowercase"]

    def transform(self, X, **fit_params):
        rx = re.compile("[^a-zA-Z]+")
        X_cleaned = [rx.sub(' ', str(t)).strip() for t in X]
        if self.lowercase:
            X_cleaned = [t.lower() for t in X_cleaned]
        return X_cleaned

    def get_params(self, deep=True):
        return self.options

    def set_params(self, **parameters):
        self.options = parameters

class ModelSummary(object):
    """The DisplayTextModel is a wrapper around `TextModel` for the
    purpose of displaying the results of the model on a web site. This
    includes labels (e.g. for the variables), but also tables that
    describe variable importance."""

    def __init__(self, text_model, data):
        self.text_model = text_model

        data['y_hat'] = text_model.predict(data.text)

        ridge = self.text_model.pipe.named_steps['ridge_model']

        self.std_X = ridge.std_X
        self.mean_outcome_in_groups = mean_outcome_in_groups(data.y, data.y_hat)

        self.percent_correct = share_correct(data.y, data.y_hat)
        self.outcome_summary = get_summary(data.y)

        self.coef = ridge.coef_
        self.number_of_features = len(self.coef)

        # Extracting the features from the featurizers. To do this, we need
        # to remove the beginning string that is added by `Pipeline`
        self.features = self.text_model.pipe.named_steps['featurizer'].get_feature_names()
        split = [tuple(f.split("__")) for f in self.features]
        groupdict = itertools.groupby(split, lambda x: x[0])

        self.features_dict = groupdict


    def get_regression_table(self):
        """Collects the data to evaluate the importance of variables"""
        regression_table = DataFrame({"beta": self.coef, "std_X": self.std_X})
        regression_table.index = self.features

        # The Effect size is the coefficient multiplied by the standard deviation, which
        # is a good measure of the overall importance of a variable.
        regression_table['beta_normalized'] = regression_table.beta * regression_table.std_X
        regression_table['effect'] = np.fabs(regression_table['beta_normalized'])

        (transformer, feature) = zip(*[f.split("__") for f in self.features])
        regression_table['transformer'] = transformer
        regression_table['feature'] = feature
        # Sorting by effect size
        return regression_table.sort_index(by='effect', ascending=False)

    def set_performance(self, outcomes_test, texts, number_sampled=40):
        y_hat_test = self.pipe.predict(texts)
        self.mean_outcome_in_groups = mean_outcome_in_groups(outcomes_test, y_hat_test)
        self.share_correct = share_correct(outcomes_test, y_hat_test)
        self.share_correct_print = round(self.share_correct, 3)
        self.texts_test_sample = get_texts_sampled(texts, number_sampled)
        self.texts_test_performance = self.get_texts_test_performance()

    def get_texts_test_performance(self):
        """Takes the sample texts that are stored with this model
        and adds their predicted value. This is used to illustrate
        the model performance with real examples"""
        texts = self.texts_test_sample
        texts_test_performance = [(texts[i], self.predict(texts[i]), i) for i in range(len(texts))]
        texts_test_performance = sorted(texts_test_performance, key=itemgetter(1))
        return texts_test_performance

    def get_features(self, text):
        """This takes a text and translates it into features. This function thus describes central logic of the TextModel"""
        x = Pipeline(self.pipe.steps[0:-1]).transform([text]).toarray()[0]
        features = Series(self.features)[Series(x) >= 1]
        tab = self.get_regression_table()
        coefficients_for_features = tab[tab.index.isin(features)]
        return coefficients_for_features

    def prediction_summary(self, text):
        summary = {}
        try:
            text = text.encode('utf-8', 'ignore')
        except UnicodeEncodeError:
            pass
        summary["predicted_value"] = self.predict(text)
        tab = self.get_regression_table()
        tab_found = self.get_features(text)
        features_plus_minus = get_features_plus_minus(tab_found)
        summary["important_features_good_and_bad"] = features_plus_minus
        return summary


def get_texts_sampled(texts, number):
    """Returns a sample of size `number` from `texts`"""
    number_of_test_texts = min(number, len(texts))
    return random.sample(texts, number_of_test_texts)


def get_features_plus_minus(tab_found):
    """Takes a list of features and return a tuple of positive
    and negative coefficients, bot nicely formatted"""
    betas = tab_found.transpose().to_dict()
    # Careful: Since we're looping over these values twice
    # we need a list, not an iterator
    betas_list = list(betas.iteritems())

    def beta_string(name, coef):
        return "%s (%s)" % (name, np.abs(round(coef['beta'], 3)))

    betas_minus = [beta_string(name, coef) for (name, coef) in betas_list if coef['beta'] < 0]
    betas_plus = [beta_string(name, coef) for (name, coef) in betas_list if coef['beta'] > 0]
    return [betas_plus, betas_minus]


def get_printable_dataframe(data):
    """This is a convenience function for django, which returns a list of lists
    This list of lists turns the index into the first column"""
    data = np.round(data, 3)
    return list(data.itertuples())


def get_summary(x):
    """Creates basic summary statistics for `x`."""
    x = pd.Series(x)
    summ = {}
    summ["min"] = round(np.min(x), 3)
    summ["max"] = round(np.max(x), 3)
    summ["mean"] = round(np.mean(x), 3)
    summ["median"] = round(x.quantile(), 3)
    summ["sd"] = round(np.std(x), 3)
    return summ


def get_cutoffs(x, num_groups=10):
    """Get the cutoffs that splits `x` into `num_groups` equally sized groups."""
    series = Series(x)
    q = series.quantile
    def perc_low(i):
        return float(i) / num_groups
    def perc_high(i):
        return float(i + 1) / num_groups
    return [(q(perc_low(i)), q(perc_high(i))) for i in range(num_groups)]

def share_correct(y, y_hat):
    """This function is only relevant for binary models. For these models, it shows the percentage of predictions
    correctly classified using the prediction y_hat. This assumes that we classify y=1 if y_hat>.5 and y=0 otherwise"""
    df = pd.DataFrame({"y": y, "y_hat": y_hat})
    df["y_classifier"] = df.y_hat > .5
    df["correctly_classified"] = df.y_classifier == df.y
    return df.correctly_classified.mean()


def mean_outcome_in_groups(y, y_hat, num_groups=10):
    """Get the average of the outcome y when y_hat is cut into num_groups equally-sized groups. This
    is used as a measure of performance of the model"""
    cutoffs = get_cutoffs(y_hat, num_groups)
    return mean_outcome_by_cutoff(y, y_hat, cutoffs)


def mean_outcome_by_cutoff(y, y_hat, cutoffs):
    """Show the average outcome y by the cutoffs for y_hat"""
    y_by_group = []
    df = pd.DataFrame({"y": y, "y_hat": y_hat})
    # Get performance from test sample (test==2), not from valdiation sample (test==1)
    for cutoff_low, cutoff_high in cutoffs:
        data_group = df[(df.y_hat >= cutoff_low) & (df.y_hat < cutoff_high)]
        y_by_group.append(np.mean(data_group["y"]))
    performance = []
    return [(i + 1, round(y_by_group[i], 3)) for i in range(len(cutoffs))]


def convert_performance_to_string(performance):
    performance_string = []
    note = ""
    for decile in performance:
        value = decile[1]
        value_string = str(value)
        if math.isnan(value):
            value_string = "N/A*"
            note = "Since deciles are determined using the training sample, it cannot be ensured that all deciles can be evaluated in the test sample"
        decile_string = (decile[0], value_string)
        performance_string.append(decile_string)
    return (performance_string, note)


def text_model_parameters(filename, train=True):
    """Given a filename (and a training flag), returns all the data needed to create a DisplayTextModel,
    which consists of the outcome data (values and name), the texts (values and name), and the
    display parameters"""
    train_string = "_train"
    if not train:
        train_string = "_test"

    filename_full = data_folder + "/" + filename + train_string + ".csv"
    descriptions_filename = "textpredictions/static/textpredictions/descriptions.json"
    descriptions = json.load(open(descriptions_filename, "r"))
    display_parameters = descriptions[filename]

    data_original = pd.read_csv(filename_full)

    outcome = display_parameters['outcome']
    text_variable = display_parameters['text_variable']

    if not outcome:
        raise (ValueError("No outcome set"))
    if not text_variable:
        raise (ValueError("No text variable set"))

    data = data_original[[outcome, text_variable]]
    data.columns = ["y", "text"]
    outcomes = data.y
    texts = data.text
    return outcomes, outcome, texts, text_variable, display_parameters


def get_similarity(text_1, text_2):
    words_1 = set(word_tokenize(text_1.lower()))
    words_2 = set(word_tokenize(text_2.lower()))
    if len(words_1) == 0 or len(words_2) == 0:
        return 0, 0

    intersection = words_1.intersection(words_2)
    num_intersection = float(len(intersection))
    return (num_intersection / len(words_1), num_intersection / float(len(words_2)))