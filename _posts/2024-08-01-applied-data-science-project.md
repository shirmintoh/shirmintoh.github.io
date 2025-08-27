---
layout: post
author: Name
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background
Provide an overview of your team's project business goals and objectives and state the objective that you are working on. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Work Accomplished
Document your work done to accomplish the outcome

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


### Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Recommendation and Analysis
Explain the analysis and recommendations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
