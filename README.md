# Mini Siri - 

Implemented a working Natural Language Processing (NLP) system, i.e., a mini Siri, using multinomial logistic regression. I then used my algorithm to extract flight information
from natural text. I did some basic feature engineering, through which I was able to improve the learner’s performance on this task.
The first model was trained based on the current word only and had an error rate of 15% on the training data, and 16% on test data.
The second model was trained on the occurrences of the current, previous, and next word. The error rate for this model was 3% on training data and 6% on test data.

Language - Python 2.7
Dependencies - NumPy


tagger1.py has the code for the logistic regression. 
The datasets are in train copy.tsv, test copy.tsv, and validation copy.tsv.
