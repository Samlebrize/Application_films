import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import scipy.stats as st
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from kneed import KneeLocator

pd.options.mode.chained_assignment = None


df = pd.read_csv("movie_metadata.csv")

pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',60)

df['long_metrage'] = 0
df['long_metrage'][df['duration'] >60] = 1

df['couleur']=0
df['couleur'][df['color']=="Color"] = 1
df.drop(columns = ['color'], inplace = True)

df.drop_duplicates(subset = ['movie_title','director_name'],keep = 'first', inplace=True)

df.isna().sum()[(df.isna().sum())<20]

df['plot_keywords'][df.plot_keywords.isna()]="Unknown"
df[['imdb_score','cast_total_facebook_likes','couleur','long_metrage','genres','plot_keywords']].isna().sum()

df.describe()

def clean_alt_list(list_):
    list_ = '["'+list_+'"]'
    list_ = list_.replace('|', '","')
    return list_

def to_1D(series):
 return pd.Series([x for _list in series for x in _list])

def dummies_genre (liste):
    for X in liste:
        searchfor=[X]
        df['genre_'+X]=df['genres'].apply(lambda x: 1 if any(i in x for i in searchfor) else 0)
        
def dummies_Keywords (liste):
    for X in liste:
        searchfor=[X]
        df['Keyword_'+X]=df['plot_keywords'].apply(lambda x: 1 if any(i in x for i in searchfor) else 0)  

df["genres"] = df["genres"].apply(clean_alt_list)
df["genres"] = df["genres"].apply(eval)

df["plot_keywords"] = df["plot_keywords"].apply(clean_alt_list)
df["plot_keywords"] = df["plot_keywords"].apply(eval)

list_keywords = to_1D(df['plot_keywords']).value_counts()
keywords_df = pd.DataFrame(data=list_keywords)
keywords_df = keywords_df[keywords_df[0]>21]

keywords_df[0] = keywords_df[0].sort_values()
list_keywords = keywords_df.head(50).index.tolist()

list_genres = to_1D(df['genres']).value_counts()
genres_df = pd.DataFrame(data=list_genres)
list_genres =genres_df.index[genres_df[0]>10].tolist()

df.reset_index(drop=True)
data = df
dummies_Keywords(list_keywords)
dummies_genre(list_genres)
df.drop(columns='Keyword_Unknown',inplace=True)

cluster = ['couleur','long_metrage']
genres = df in df.columns.str.contains('genre_')==True
genres=df.loc[:,genres].columns.tolist()
cluster.extend(genres)

keywords = df in df.columns.str.contains('Keyword_')==True
keywords=df.loc[:,keywords].columns.tolist()
cluster.extend(keywords)

df_cluster = df.loc[:,cluster]


X=df_cluster.to_numpy()

std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

#Test train split for supervised training
X_train,X_test=train_test_split(X_scaled,test_size=0.5)


data.drop(columns=cluster,inplace=True)



kmeans_kwargs = {
    "init": "random",
    "n_init": 40,
    "max_iter": 1000,}


# A list holds the SSE values for each k
sse = []
for k in range(1, 200):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X_train)
    sse.append(kmeans.inertia_)
    

kneedle = KneeLocator(range(1, 200), sse, S=1, curve="convex", direction="decreasing",online = True, interp_method="polynomial")

best_kmeans = KMeans(n_clusters=kneedle.elbow, **kmeans_kwargs)
best_kmeans.fit(X_train)



std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)
clustering = best_kmeans.fit_predict(X)
df_cluster['cluster'] = clustering.tolist()

clusters_too_small = df_cluster['cluster'].value_counts()[df_cluster['cluster'].value_counts().values<10].index.tolist()
#clusters_too_small

df_cluster['cluster'][df_cluster['cluster'].isin(clusters_too_small)]=np.nan

df_cluster_to_adjust = df_cluster[df_cluster['cluster'].notna()]

X=df_cluster_to_adjust.loc[:,cluster]
y=df_cluster_to_adjust['cluster']

#Test train split for supervised training
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

param_grid={'n_neighbors':np.arange(1,20)}
grid=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
grid.fit(X_train,y_train)

class_film=grid.best_estimator_

df_cluster['cluster_rec'] = class_film.predict(df_cluster.loc[:,cluster])
df_rec = df_cluster['cluster'].isna()
df_cluster.loc[df_rec,'cluster']=df_cluster.loc[df_rec,'cluster_rec']
df_cluster.drop(columns='cluster_rec', inplace = True)

data['cluster'] = df_cluster['cluster']
data.rename(columns={"index": "Id"})

data.to_csv()

df_m = data
df_m = df_m.reset_index(drop=False)
df_marks = df_m[['index','movie_title']]
df_marks
html = df_marks.to_html(index=0)
#write html to file
text_file = open("Id_to_movie_titles.html", "w")
text_file.write(html)
text_file.close()

Print('Terminated')