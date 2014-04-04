The goal is to write a package to simplify text analysis with python. 
This should include both feature extraction and building predictive models.

Here's how I think it should work: Assume that you have a list of texts. Then you should be 
able to write get_features(texts,type="bag-of-words") to extract word counts, and you should be able
to write get_features(texts,type="topics") to extract the topics contained in the text.

Now assume that you also have a vector of outcomes. To obtain a predictive 
model from the text, you write model=TextModel(texts,outcomes,type="bag-of-words").
If you want to know the outcome for some new text, you would write
predicted_value=model.predict("Some new text").

Simplifying the model this far does of course require making lots of assumptions
along the way. If you want to, you should be able change these defaults one-by-one.

Optimally, it would be easy to treat different types of features as 
modules that can be turned on and off, for instance by using type=["bag-of-words","topic"]