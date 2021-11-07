"""
The following example is to cluster documents
Reimplemented based on reference: http://brandonrose.org/clustering
"""
import nltk, re
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")   # initiate snowball stemmer from nltk

# this function overrides tokenizer for TfidfVectorizer in sklearn
# input is a piece of text
# output is a list of tokens
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# get the raw text of the movie reviews
# documents is a list of string
documents = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
# use the TfidfVectorizer in the sklearn package to transform a corpus in vector space model
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, tokenizer=tokenize_and_stem)
# obtain the vector space matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
# k for k-means
k = 6
# initiate KMeans model
km = KMeans(n_clusters=k)
# use KMeans model to fit the vector space representation of the corpus
km.fit(tfidf_matrix)
# obtain the cluster assignment for the documents as a list
clusters = km.labels_.tolist()
# print out the cluster membership of each document
print(list(zip(movie_reviews.fileids(), clusters)))
# obtain centroids of the clusters with features sorted from largest value to smallest value
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
# get the terms
terms = tfidf_vectorizer.get_feature_names()
# print top terms for each cluster
for i in range(k):
    print("Cluster", i, ":")
    print([terms[ind] for ind in order_centroids[i, :15]])
