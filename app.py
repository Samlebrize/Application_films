import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import random
import flask
from sklearn import preprocessing 

data = pd.read_csv("metadata.csv")

voisin = ['imdb_score','cast_total_facebook_likes','cluster']
df_distance = data.loc[:,voisin]
rnd = random.Random()


X=df_distance
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)
X=pd.DataFrame(X_scaled,columns=df_distance.columns)

def get_recommendations(idfilm):
    idfilm = int(idfilm)
    tmp_X = X.sub(X.loc[idfilm], axis='columns')
    tmp_X = X[X['cluster'] == X.loc[idfilm,'cluster']]
    tmp_series = tmp_X.apply(np.square).apply(np.sum, axis=1).sort_values()
    rad = data.loc[tmp_series.index].head(20)
    filmor = data.loc[idfilm]
    if idfilm in rad.index:
        rad.drop(index = idfilm, inplace = True),
    movies = rad.loc[np.random.choice(rad.index, 5, replace=False)]
    movies = movies.append(data.loc[idfilm])
    movies = movies[::-1]
    movies = movies.reset_index(drop=False)
    movies.rename(columns={"index": "Id"},inplace=True)
    movies = movies.loc[:,['Id','movie_title']]
    type_film = ['Référence :','Recommandé 1 :','Recommandé 2 :',
                 'Recommandé 3 :','Recommandé 4 :','Recommandé 5 :']
    movies.insert(0, 'films', type_film)
    data_titles = movies
    return data_titles


app = flask.Flask(__name__, template_folder='templates')
app.config["DEBUG"] = True

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_id = flask.request.form['id_film']
        result_final = get_recommendations(m_id)
        films = []
        id_film = []
        movie_tit = []

        for i in range(len(result_final)):
            films.append(result_final.iloc[i][0])
            id_film.append(result_final.iloc[i][1])
            movie_tit.append(result_final.iloc[i][2])

        return flask.render_template('resultats.html',movie_type=films,movie_id=id_film,movie_title=movie_tit,search_name=m_id)


@app.route('/Id_to_movie_titles.htmL')
def liste_id():
    return flask.render_template('Id_to_movie_titles.html')


if __name__ == '__main__':
    app.run()