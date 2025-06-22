import re
import string
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin

def clean_essay(essay):
    if pd.isna(essay):
        return ""
    essay = essay.lower()
    essay = re.sub(r'\d+', '', essay)
    essay = re.sub(r'\b\w*\d\w*\b', '', essay)
    essay = re.sub(r'[^a-zA-Z\s]', ' ', essay)
    essay = re.sub(r'\s+', ' ', essay)
    essay = essay.strip()
    return essay

def word_removal(essay):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(essay)
    filtered_sentence = [
        w for w in word_tokens
        if w not in stop_words and len(w) > 1 and w.strip() not in string.punctuation
    ]
    return filtered_sentence

def wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatized_tokens(tokens):
    tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, wordnet_pos(tag)) for word, tag in tagged]
    return lemmatized

class EssayPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.preprocess_essay)

    def preprocess_essay(self, essay):
        cleaned_essay = clean_essay(essay)
        tokens = word_removal(cleaned_essay)
        lemmatized = lemmatized_tokens(tokens)
        return ' '.join(lemmatized)