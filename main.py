import math
import operator

import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import islice
import os

if __name__ == '__main__':
    print("Empieza la ejecucion")
    id_usuario=int (input("Selecciona el id del usuario activo "))

    # Get the current working directory
    current_directory = os.getcwd()

    # Cargar datos de los archivos CSV en pandas DataFrames
    usuarios = pd.read_csv(os.path.join(current_directory, "users.csv"),header=None)
    usuarios.columns=['id_usuario','nombre']
    peliculas = pd.read_csv(os.path.join(current_directory, "movie-titles.csv"),header=None)
    peliculas.columns=['id_item','titulo']
    valoraciones = pd.read_csv(os.path.join(current_directory, "ratings.csv"),header=None)
    valoraciones.columns=['id_usuario','id_item','valoracion']
    etiquetas = pd.read_csv(os.path.join(current_directory, "movie-tags.csv"),header=None,encoding='latin-1')

    # Establecer nombres de columnas para el DataFrame
    etiquetas.columns=['id_item','etiqueta']
    valoraciones_usuario = valoraciones[(valoraciones['id_usuario'] == id_usuario)]
    etiquetas_tf_usuario = []
    tf_peliculas = []
    perfiles_producto = []

    # Crear una lista de etiquetas únicas (etiquetas)
    etiquetas_distintas=etiquetas['etiqueta'].unique()

    # Cree una lista de ID de películas únicas (id_item)
    peliculas_distintas = peliculas['id_item'].unique()

    # Inicializar diccionarios para frecuencias de términos (tfs), tf-idf (tfs_idf) y tf-idf normalizado (tfs_idf_u)    tfs = {}
    tfs = {}
    tfs_idf = {}
    tfs_idf_u = {}

    # Iterar sobre cada película en el DataFrame de 'peliculas'
    for a in peliculas.itertuples():
        # Inicializar diccionarios para la película actual
        tfs[a.id_item] = {}
        tfs_idf[a.id_item] = {}
        tfs_idf_u[a.id_item] = {}

        # Inicializar frecuencias de términos para cada etiqueta
        for b in etiquetas_distintas:
            tfs[a.id_item][b] = 0
            tfs_idf[a.id_item][b] = 0
            tfs_idf_u[a.id_item][b] = 0

    # Calcular frecuencias de términos para cada película en función de las etiquetas en el DataFrame 'etiquetas'
    for etiqueta in etiquetas.itertuples():
        id_pelicula=etiqueta.id_item
        if not etiqueta.id_item in tfs:
            tfs[etiqueta.id_item] = {}
            tfs[etiqueta.id_item][etiqueta.etiqueta]=0
        elif not etiqueta.etiqueta in tfs[etiqueta.id_item]:
            tfs[etiqueta.id_item][etiqueta.etiqueta] = 1
        else:
            tfs[etiqueta.id_item][etiqueta.etiqueta]+=1

    # Calcular frecuencias de documentos inversas (idfs) e inicialice el perfil de usuario
    idfs = {}
    perfil_usuario = {}
    for etiqueta in etiquetas_distintas:
        perfil_usuario[etiqueta]=0

        # Filtrar el DataFrame de 'etiquetas' para obtener películas con la etiqueta actual
        filtrado = etiquetas[etiquetas['etiqueta']==etiqueta]
        filtrado=filtrado['id_item'].unique()

        # Calcular el valor idf para la etiqueta actual
        idfs[etiqueta]=math.log(len(peliculas_distintas) / len(filtrado),10)

    # Calcule los valores tf-idf y los valores tf-idf normalizados para cada película
    for a in tfs:
        aux = list(tfs[a].values())
        aux2 = np.asarray(aux)
        norma = np.linalg.norm(aux2)
        for b in tfs[a]:
            tfs_idf[a][b] = tfs[a][b] * idfs[b]
            tfs_idf_u[a][b] = (tfs[a][b]*idfs[b])/norma
    for a in tfs_idf:
        aux = list(tfs_idf[a].values())
        aux2 = np.asarray(aux)
        norma = np.linalg.norm(aux2)
        for b in tfs_idf[a]:
            tfs_idf_u[a][b] = (tfs[a][b] * idfs[b]) / norma

    # Calcular la calificación media para el usuario.
    media = valoraciones_usuario['valoracion'].mean()
    contador=range(len(valoraciones_usuario.values))
    vector_aux=valoraciones_usuario.values
    for a in etiquetas_distintas:
        perfil_usuario[a]=0

        # Calcular la suma ponderada de los valores tf-idf normalizados para el perfil del usuario
        for b in contador:
            id_item=vector_aux[b][1]
            valor=vector_aux[b][2]
            perfil_usuario[a]+=tfs_idf_u[id_item][a]*(valor-media)

    # Calcular la similitud de coseno entre el perfil del usuario y cada película
    similitud = {}
    ids_peliculas_valoradas=np.array(valoraciones_usuario.values.transpose()[1])
    vector_1 = list(perfil_usuario.values())
    v1 = np.asarray(vector_1)
    p1 = v1.reshape(1, -1)
    for pelicula in peliculas_distintas:
        if pelicula not in ids_peliculas_valoradas:
            vector_2 = list(tfs_idf_u[pelicula].values())
            v2 = np.asarray(vector_2)
            p2 = v2.reshape(1, -1)
            similitud[pelicula] = cosine_similarity(p1,p2)
    
    # Ordenar el diccionario de similitud en orden descendente
    similitudes = dict(sorted(similitud.items(),key=operator.itemgetter(1),reverse=True))
    
    # Conviertir el diccionario de similitud en una matriz numpy
    recomendaciones = np.asarray(similitudes.items())

    # Imprimir el top 10 de recomendaciones para el usuario seleccionado
    print("")
    print('Top 10 recomentadions for user '+str(id_usuario))
    for key, value in islice(recomendaciones.item(),10):
        print(peliculas.loc[peliculas['id_item'] == key, 'titulo'].values[0], ": ",value[0][0])
