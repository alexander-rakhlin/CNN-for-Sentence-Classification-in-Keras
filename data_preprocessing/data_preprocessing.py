import pandas as pd

data_name = "test_set"

# read the data
reviews_df = pd.read_csv('../data/clothing_review.csv')

# Remove last 500 Rows  and save as test set
# reviews_df.tail(500).to_csv('test_set.csv')
# reviews_df.drop(reviews_df.tail(500).index, inplace=True)  # drop last n rows
reviews_df = reviews_df.tail(500)  # test set

# Let's check out the missing values
reviews_df.isnull().sum()  # Let's check out the missing values
reviews_df.isnull().sum()

# drop rows with NA
reviews_df.dropna(subset=['Review Text', 'Division Name'], inplace=True)
reviews_df['Rating'].value_counts()

# clean reviews, removing everything except alphabets
# Remove Punctuation
reviews_df['Review_Tidy'] = reviews_df['Review Text'].str.replace("[^a-zA-Z#]", " ")

# Stop Words Removal
from nltk.corpus import stopwords

stop = stopwords.words('english')

# Code from https://www.kaggle.com/pjoshi15/so-many-outfits-so-little-time-word2vec
reviews_df['Review_Tidy'] = reviews_df['Review_Tidy'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Combining title and review
reviews_df["Review_Tidy"] = reviews_df["Title"].map(str) + " " + reviews_df["Review_Tidy"]

# Lementizing
# https://pythonprogramming.net/lemmatizing-nltk-tutorial/
print("Lemmatizing....")
from nltk import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
for review in reviews_df['Review_Tidy']:
    for word in review:
        word = lemmatizer.lemmatize(word)

# Remove Repeated Characters
print("Removing repeated Characters....")
import re

for review in reviews_df['Review_Tidy']:
    for word in review:
        word = re.sub(r'(.)\1+', r'\1\1', word)

# Covert to lowercase
print("Converting words to lowercase...")
reviews_df['Review_Tidy'] = reviews_df['Review_Tidy'].str.lower()

# Fix misspelled Words
print("Fixing misspelled words...")
from spellchecker import SpellChecker

spell = SpellChecker()
# https://github.com/barrust/pyspellchecker
for review in reviews_df['Review_Tidy']:
    for word in review:
        word = spell.correction(word)

print(reviews_df.head())
pos_reviews = []
neg_reviews = []

print("Creating Pos and Neg Lists...")
for index, review in reviews_df.iterrows():
    if review[5] >= 3:
        pos_reviews.append(review['Review_Tidy'] + '\n')
    else:
        neg_reviews.append(review['Review_Tidy'] + '\n')

print("Writing to File")
for review in pos_reviews:
    with open(data_name + '.pos', "a") as file:
        file.write(review)

for review in neg_reviews:
    with open(data_name + '.neg', "a") as file:
        file.write(review)
