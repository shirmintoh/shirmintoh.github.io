---
layout: post
author: Toh Shir Min
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background
Perform topic modelling to uncover insights on the topics prevalent in movies with (neutral / positive) sentiment and likewise for movies with negative sentimentto determine box office success rate using IDMB & TMDB Movie Metadata Big Dataset 

### Data Preparation
# obtain information about dataframe
print(df.head())
print()
print(df.info())# obtain information about dataframe
print(df.head())
print()
print(df.info())
# To generate a summary table of the dataframe
def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['%missing'] = df.isnull().sum().values / len(df) * 100
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['average'] = desc['mean'].values
    summ['standard_deviation'] = desc['std'].values

    return summ
    summary(df).style.background_gradient(cmap='YlOrBr')

### Modelling
#To observe the distribution of all numeric data
df.hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()
%matplotlib inline
# Distribution of movies/shows by production countries (top 50)
plt.figure(figsize=(15, 6))  
df['production_countries'].value_counts().head(50).plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number of Movies')
plt.show()
# Distribution for release year
df['release_year'].plot.hist(bins=15, alpha=0.5)
# Generare Title + Year to verify uniqueness (some of the movie title may be remakes)
df['title_year'] = df['title'] + df['release_year'].astype(str)

# Remove duplicated rows
df.drop_duplicates(subset=['title_year'], inplace=True)
df.shape
# Remove rows where runtime = 0 (logical exclusion)
df = df[~df[['runtime']].isin([0]).any(axis=1)]
df.shape
# Noticed a substantial number of rows containing value = 0 in columns "revenue" and "budget"
print(df[df['revenue'] == 0].shape[0])
print(df[df['budget'] == 0].shape[0])
# Dropping columns "revenue" and "budget" as these column a lot of missing information (i.e, 0 is null entry)
df = df.drop('revenue', axis=1)
df = df.drop('budget', axis=1)
# Dropping these columns due to too much missing information
df = df.drop('IMDB_Rating', axis=1)
df = df.drop('Meta_score', axis=1)
df = df.drop('Certificate', axis=1)
df = df.drop('Star1', axis=1)
df = df.drop('Star2', axis=1)
df = df.drop('Star3', axis=1)
df = df.drop('Star4', axis=1)
df = df.drop('Music_Composer', axis=1)
df = df.drop('Director_of_Photography', axis=1)
df = df.drop('Producers', axis=1)
df = df.drop('production_companies', axis=1)
df = df.drop('Writer', axis=1) 
df = df.drop('tagline', axis=1)

Data transformation and feature engineering
# choose relevant remaining columns for our topic modelling
columns_to_keep  = ["overview", "keywords", "genres_list", "all_combined_keywords","overview_sentiment"]
df_mod = df[columns_to_keep]
# Since overview is an important feature to determine to overview sentiment of a movie, remove rows containing no movie overview
df_mod = df_mod[df_mod['overview'].notna()]
df_mod.shape
# Combining all text columns
texts = df_mod[["overview", "keywords", "genres_list", "all_combined_keywords"]].fillna("").agg(" ".join, axis=1)


# Merge with original DataFrame
df_combined = pd.concat([df_mod, texts], axis=1)

# Drop old columns, keep only 'combined_text'
columns_to_drop = ["overview", "keywords", "genres_list", "all_combined_keywords"]
df_mod2 = df_combined.drop(columns=columns_to_drop)

print(df_mod2.head())
# Create a new column "Sentiment". where 0 or more than 0 is "not negative", while less than 0 is "negative" 
df_mod2['sentiment'] = df_mod2['overview_sentiment'].apply(
    lambda x: 'not negative' if x >= 0 else 'negative'
)

df_mod2 = df_mod2.drop('overview_sentiment', axis=1)
df_mod2.columns = ['text', 'sentiment']
value_counts = df_mod2["sentiment"].value_counts()
print(value_counts)
print(df_mod2.shape)
df_mod2.head()

Pre-process Text Data
# My data contains about ~700k rows, using python loops and NLTK functions is too slow, hence I use a vectorized and faster approach
import swifter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Initialize
stop_words = set(stopwords.words("english"))
exclude_punct = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

# Create a translation table for removing punctuation
translator = str.maketrans('', '', string.punctuation)

# Fast preprocessing function
def preprocess_text(text):
    text = str(text).lower()                     # lowercase
    text = text.translate(translator)            # remove punctuation
    tokens = text.split()                        # split by whitespace
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df_mod2["text_clean"] = df_mod2["text"].astype(str).swifter.apply(preprocess_text)
df_mod2 = df_mod2.drop("text",axis=1)
# Check for frequently appearing word
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_mod2["text_clean"].astype(str))

# Sum occurrences of each word
word_sum = X.sum(axis=0)
words_freq = [(word, word_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Top 20 frequent words
print(words_freq[:20])
# Extract top 20
top_words = words_freq[:20]
words, counts = zip(*top_words)  # unzip into two lists

# Plot horizontal axis with words
plt.figure(figsize=(14,6))
sns.barplot(x=list(words), y=list(counts), palette="viridis")
plt.ylabel("Frequency")
plt.xlabel("Word")
plt.title("Top 20 Most Frequent Words")
plt.xticks(rotation=45, ha="right") 
plt.show()
#One more round cleaning after updating stop words
#Remove common word
custom_stopwords = {'unknown', 'film', 'documentary','drama', 'life'}  #by logical exclusion
stop_words.update(custom_stopwords)

df_mod2["text_clean"] = df_mod2["text_clean"].astype(str).swifter.apply(preprocess_text)
# Split data set into not_negative and negative sentiment
df_not_negative = df_mod2[df_mod2["sentiment"] == 'not negative']
df_negative = df_mod2[df_mod2["sentiment"] == 'negative']
# Creating bag of words for NOT negative sentiment
from gensim import corpora
from gensim.utils import simple_preprocess

# Tokenize using gensim's simple_preprocess
df_not_negative.loc[:, "tokens"] = df_not_negative["text_clean"].apply(lambda x: simple_preprocess(x, deacc=True))

# Create a Gensim dictionary
dictionary = corpora.Dictionary(df_not_negative["tokens"])

# Filter extremes
# Since we have ~700k rows, creating gensim dictionary and BOW requires huge memory, I have to increase filtering extremes
# no_below: ignore words that appear in less than 5 documents
# no_above: ignore words that appear in more than 50% of documents
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Create Bag of Words representation for each document, creating BOW safely with .loc
df_not_negative.loc[:, "bow"] = df_not_negative["tokens"].apply(
    lambda tokens: dictionary.doc2bow(tokens) 
)

print(df_not_negative[["text_clean", "tokens", "bow"]].head(10))
# Creating bag of words for negative sentiment
from gensim import corpora
from gensim.utils import simple_preprocess

# Tokenize using gensim's simple_preprocess
df_negative.loc[:, "tokens"] = df_negative["text_clean"].apply(lambda x: simple_preprocess(x, deacc=True))

# Create a Gensim dictionary
dictionary = corpora.Dictionary(df_negative["tokens"])

# Filter extremes
# Since we have ~700k rows, creating gensim dictionary and BOW requires huge memory, I have to increase filtering extremes
# no_below: ignore words that appear in less than 5 documents
# no_above: ignore words that appear in more than 50% of documents
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Create Bag of Words representation for each document
df_negative.loc[:, "bow"] = df_negative["tokens"].apply(
    lambda tokens: dictionary.doc2bow(tokens)
)

print(df_negative[["text_clean", "tokens", "bow"]].head(10))

LDA models
# NOT negative
# Running full >700k rows of data for my LDA model is not prudent, hence I am sampling only 10% of data for this purpose

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel

# Randomly sample 10% of data
df_sample_not_negative = df_not_negative.sample(frac=0.1, random_state=123).reset_index(drop=True)
print(f"Sample size: {len(df_sample_not_negative)} rows")
# Prepare dictionary
dictionary = Dictionary(df_sample_not_negative['tokens'])
# filter extremes
dictionary.filter_extremes(no_below=3, no_above=0.5)
# Save dictionary
dictionary.save('dictionary_sample.dict')
# Convert tokens to BOW & serialize
df_sample_not_negative['bow'] = df_sample_not_negative['tokens'].apply(lambda x: dictionary.doc2bow(x))
# Serialize BOW to disk (memory-efficient)
MmCorpus.serialize('bow_corpus_sample.mm', df_sample_not_negative['bow'])
bow_corpus_sample = MmCorpus('bow_corpus_sample.mm')

#LDA model
lda_model_note_negative = LdaModel(
    corpus=bow_corpus_sample,
    id2word=dictionary,
    num_topics=10,
    passes=5,          
    chunksize=10000,
    update_every=1,    # online learning
    alpha='auto',
    eta='auto',
    random_state=42
)


# Save final model
lda_model_note_negative.save('lda_model_not_negative.model')
print("LDA model trained on 10% sample and saved.")
# View topics (NOT negative)
num_words = 5
for idx, topic in lda_model_note_negative.print_topics(num_topics=10, num_words=num_words):
    print(f"Topic {idx}: {topic}\n")
    # Computing Coherence score for not negative sentiment 
from gensim.models import CoherenceModel

coherence_model_not_neg = CoherenceModel(
    model=lda_model_note_negative,
    texts=df_sample_not_negative['tokens'],  
    dictionary=dictionary,
    coherence='c_v'
)

coherence_score = coherence_model_not_neg.get_coherence()
print(f"Coherence Score: {coherence_score:.4f}")

Coherence Score: 0.4888

# Negative
# 10% of data being used

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel

# Randomly sample 10% of data
df_sample_negative = df_negative.sample(frac=0.1, random_state=123).reset_index(drop=True)
print(f"Sample size: {len(df_sample_negative)} rows")
# Prepare dictionary
dictionary_neg = Dictionary(df_sample_negative['tokens'])
# filter extremes
dictionary_neg.filter_extremes(no_below=3, no_above=0.5)
# Save dictionary
dictionary_neg.save('dictionary_sample_negative.dict')
# Convert tokens to BOW & serialize
df_sample_negative['bow'] = df_sample_negative['tokens'].apply(lambda x: dictionary_neg.doc2bow(x))
# Serialize BOW to disk (memory-efficient)
MmCorpus.serialize('bow_corpus_sample_neg.mm', df_sample_negative['bow'])
bow_corpus_sample_neg = MmCorpus('bow_corpus_sample_neg.mm')

#LDA model
lda_model_negative = LdaModel(
    corpus=bow_corpus_sample_neg,
    id2word=dictionary_neg,
    num_topics=10,
    passes=5,          
    chunksize=10000,
    update_every=1,    # online learning
    alpha='auto',
    eta='auto',
    random_state=42
)


# Save final model
lda_model_negative.save('lda_model_negative.model')
print("LDA model trained on 10% sample and saved.")

Sample size: 18732 rows
LDA model trained on 10% sample and saved.

# View topics (negative)
num_words = 5
for idx, topic in lda_model_negative.print_topics(num_topics=10, num_words=num_words):
    print(f"Topic {idx}: {topic}\n")

    # Computing Coherence score for negative sentiment 
from gensim.models import CoherenceModel

coherence_model_neg = CoherenceModel(
    model=lda_model_negative,
    texts=df_sample_negative['tokens'],  
    dictionary=dictionary_neg,
    coherence='c_v'
)

coherence_score_neg = coherence_model_neg.get_coherence()
print(f"Coherence Score: {coherence_score_neg:.4f}")
Coherence Score: 0.3997

### Evaluation
Visualizing topics and key words
# NOT negative
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook() 
vis = gensimvis.prepare(lda_model_note_negative, df_sample_not_negative["bow"].tolist(), dictionary)

vis

# negative
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook()  
vis_neg = gensimvis.prepare(lda_model_negative, df_sample_negative["bow"].tolist(), dictionary_neg)

vis_neg


## Recommendation and Analysis
Successfully identified at least 2 topics that are associated with neutral / positive sentiment
Movies about love and family
Horror and thriller

Also able to identify some topics that gave negative sentiment
Drug, crime, politics
Vulgarities

Some topics appear in both groups 
Such as war and family (but their contexts are different)
It is very clear that only vulgar topics mainly appear in negative sentiment

Areas for improvement
Domain specific customizations- Custom stop words or custom dictionaries. Still had many domain specific words appearing such as “woman” and “man”. Might be good to have custom stop words that are inline with movie reviews.

Include bigram and trigram models might be able to give more meaningful insights rather than just basing on single words 

Separate my data into 3 groups such as negative, neutral and positive instead of two groups (not-negative) and negative


## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
