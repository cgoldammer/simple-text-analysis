from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nameparser.parser import HumanName
from pprint import pprint
import nltk
nltk.data.path.append('/Users/cg/Dropbox/code/Python/nltk_data/')

tokens=["James","yes","IBM","GM"]
pos = nltk.pos_tag(tokens)
pos
sentt = nltk.ne_chunk(pos, binary = False)
sentt



text = """
My dad is John and he is nice.
In Spain and England and New York Some economists have responded positively to Bitcoin, 
including 
Francois R. Velde, senior economist of the Federal Reserve in Chicago 
who described it as "an elegant solution to the problem of creating a 
digital currency." In November 2013 Richard Branson announced that 
Virgin Galactic would accept Bitcoin as payment, saying that he had invested 
in Bitcoin and found it "fascinating how a whole new global currency 
has been created", encouraging others to also invest in Bitcoin.
Other economists commenting on Bitcoin have been critical. 
Economist Paul Krugman has suggested that the structure of the currency 
incentivizes hoarding and that its value derives from the expectation that 
others will accept it as payment. Economist Larry Summers has expressed 
a "wait and see" attitude when it comes to Bitcoin. Nick Colas, a market 
strategist for ConvergEx Group, has remarked on the effect of increasing 
use of Bitcoin and its restricted supply, noting, "When incremental 
adoption meets relatively fixed supply, it should be no surprise that 
prices go up. And that’s exactly what is happening to BTC prices."
"""

pos = nltk.pos_tag(text.split(" "))
pos
sentt = nltk.ne_chunk(pos, binary = False)
sentt

for w in sentt:
    print w

names = named_entities(text)
names


        