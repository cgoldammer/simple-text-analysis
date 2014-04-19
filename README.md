The goal is to write a package to simplify text analysis with python. 
This should include both feature extraction and building predictive models.

The package passes simple unit tests, but requires additional testing.

To use the package, you need data: A list of outcomes and texts. 
To obtain a predictive bag-of-words model from the text, you write 

```python
from text_model_functions import TextModel
modules='bag-of-words'
text_model=TextModel(outcomes,texts,modules)
```

If you want to know the predicted value for new text, you write

```python
text_model.predict("Some new text")
```
        
The package also contains additional feature extractors, for instance 
emotions (positive/negative and subjective/objective) and named entities 
(people and organizations). To extract these as well, you would write:
        
```python
modules=['bag-of-words','emotions','entities']
text_model=TextModel(outcomes,texts,modules)
```

The package does the text cleaning for you. But you can also change 
default options, for instance by setting:
        
```python
options={'lowercase':True,'lemmatize':True}
text_model=TextModel(outcomes,texts,modules,options)
```

Simplifying the model this far does of course require making lots of assumptions
along the way. If you want to, you should be able change these defaults one-by-one.
