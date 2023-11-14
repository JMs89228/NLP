import pandas as pd
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

import string
from operator import itemgetter
from collections import Counter, OrderedDict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

wine_df = pd.read_csv('winemag-data-130k-v2.csv')

# Tokenize the sentences
reviews_list = list(wine_df['description'])
reviews_list = [str(r) for r in reviews_list]
full_corpus = ''.join(reviews_list)
sentenses_tokenized = sent_tokenize(full_corpus)

# Normalize the text
stop_words = set(stopwords.words('english')) 

punctuation_table = str.maketrans({key: None for key in string.punctuation})
sno = SnowballStemmer('english')

def normalize_text(raw_text):
    try:
        word_list = word_tokenize(raw_text)
        normalized_sentence = []
        for w in word_list:
            try:
                w = str(w)
                lower_case_word = str.lower(w)
                stemmed_word = sno.stem(lower_case_word)
                no_punctuation = stemmed_word.translate(punctuation_table)
                if len(no_punctuation) > 1 and no_punctuation not in stop_words:
                    normalized_sentence.append(no_punctuation)
            except:
                continue
        return normalized_sentence
    except:
        return ''

# sentence_sample = sentences_tokenized[:10]
normalized_sentences = []
for s in sentenses_tokenized:
    normalized_text = normalize_text(s)
    normalized_sentences.append(normalized_text)

# Extract the relevant words
phrases = Phrases(normalized_sentences)
phrases = Phrases(phrases[normalized_sentences])

ngrams = Phraser(phrases)

phrased_sentences = []
for sent in normalized_sentences:
    phrased_sentence = ngrams[sent]
    phrased_sentences.append(phrased_sentence)

full_list_words = [item for sublist in phrased_sentences for item in sublist]

# word_counts = Counter(full_list_words)
# sorted_counts = OrderedDict(word_counts.most_common(5000))
# counter_df = pd.DataFrame.from_dict(sorted_counts, orient='index')
# counter_df.to_csv('top_5000_descriptors.csv')

descriptor_mapping = pd.read_csv('descriptor_mapping.csv').set_index('raw descriptor')

# Map the descriptors
def return_mapped_descriptor(word):
    if word in list(descriptor_mapping.index):
        normalized_word = descriptor_mapping['level_3'][word]
        return normalized_word
    else:
        return word

normalized_sentences = []
for sent in phrased_sentences:
    normalized_sentence = []
    for word in sent:
        normalized_word = return_mapped_descriptor(word)
        normalized_sentence.append(str(normalized_word))
    normalized_sentences.append(normalized_sentence)

# Train the word2vec model
wine_word2vec_model = Word2Vec(normalized_sentences, vector_size=300, min_count=5, epochs=15)
print(wine_word2vec_model)

wine_word2vec_model.save('wine_word2vec_model.bin')


# From word2vec to sentence2vec
wine_reviews = list(wine_df['description'])

def return_descriptor_from_mapping(word):
    if word in list(descriptor_mapping.index):
        descriptor_to_return = descriptor_mapping['level_3'][word]
        return descriptor_to_return

descriptorized_reviews = []
for review in wine_reviews:
    normalized_review = normalize_text(review)
    phrased_review = ngrams[normalized_review]
    descriptors_only = [return_descriptor_from_mapping(word) for word in phrased_review]
    no_nones = [str(d) for d in descriptors_only if d is not None]
    descriptorized_review = ' '.join(no_nones)
    descriptorized_reviews.append(descriptorized_review)

vectorizer = TfidfVectorizer()
X = vectorizer.fit(descriptorized_reviews)

dict_of_tfidf_weightings = dict(zip(X.get_feature_names_out(), X.idf_))

wine_review_vectors = []
for d in descriptorized_reviews:
    descriptor_count = 0
    weighted_review_terms = []
    terms = d.split(' ')
    for term in terms:
        if term in dict_of_tfidf_weightings.keys():
            tfidf_weighting = dict_of_tfidf_weightings[term]
            word_vector = wine_word2vec_model.wv.get_vector(term).reshape(1, 300)
            weighted_word_vector = tfidf_weighting * word_vector
            weighted_review_terms.append(weighted_word_vector)
            descriptor_count += 1
        else:
            continue
    try:
        review_vector = sum(weighted_review_terms)/len(weighted_review_terms)
    except:
        review_vector = []
    vector_and_count = [terms, review_vector, descriptor_count]
    wine_review_vectors.append(vector_and_count)

wine_df['normalized_descriptors'] = list(map(itemgetter(0), wine_review_vectors))
wine_df['review_vector'] = list(map(itemgetter(1), wine_review_vectors))
wine_df['descriptor_count'] = list(map(itemgetter(2), wine_review_vectors))

wine_df.reset_index(inplace=True)
wine_df.to_csv('wine_df.csv')

