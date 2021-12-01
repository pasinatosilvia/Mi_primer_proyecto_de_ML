from scipy.stats import iqr
import numpy as np

def outliers_quantie(df, feature, param=1.5):
    '''
    Función que quita los outliers de la columna dada
    
    ''' 
        
    iqr_ = iqr(df[feature], nan_policy='omit')
    q1 = np.nanpercentile(df[feature], 25)
    q3 = np.nanpercentile(df[feature], 75)
    
    th1 = q1 - iqr_*param
    th2 = q3 + iqr_*param
    
    return df[(df[feature] >= th1) & (df[feature] <= th2)].reset_index(drop=True)


def decodificar_valores(x):
    '''
    Función que asigna un nombre a cada valor numérico único de un dataframe
    partiendo del diccionario de las variedades de vinos
    
    x= valor a definir segun la leyenda del diccionario 
    num_to_string = diccionario
    
    NOTA: si lo que se tiene es el nombre y se le quiere asignar el número correspondiente,
    solo hay que invertir val y pos dentro del bloque if
    '''
    num_to_string = {'Pinot Noir': 0, 'Chardonnay': 1, 'Red Blend': 2, 'Cabernet Sauvignon': 3, 
                     'Bordeaux-style Red Blend': 4, 'Syrah': 5, 'Riesling': 6, 'Malbec': 7, 
                     'Rosé': 8, 'Tempranillo': 9, 'Nebbiolo': 10, 'Sauvignon Blanc': 11, 
                     'Zinfandel': 12, 'White Blend': 13, 'Rhône-style Red Blend': 14, 
                     'Sangiovese': 15, 'Merlot': 16, 'Pinot Gris': 17, 'Cabernet Franc': 18, 
                     'Gamay': 19, 'Sparkling Blend': 20, 'Tempranillo Blend': 21, 'Gewürztraminer': 22, 
                     'Shiraz': 23, 'Viognier': 24, 'Grenache': 25, 'Champagne Blend': 26, 
                     'Rhône-style White Blend': 27, 'Petite Sirah': 28, 'Chenin Blanc': 29, 
                     'Barbera': 30, 'Garnacha': 31, 'Melon': 32, 'Bordeaux-style White Blend': 33, 
                     'Albariño': 34, 'Pinot Grigio': 35, 'Torrontés': 36, 'Pinot Blanc': 37, 'Glera': 38, 
                     'Tinta de Toro': 39, "Nero d'Avola": 40, 'Verdejo': 41, 'Petit Verdot': 42, 
                     'Mourvèdre': 43, 'Aglianico': 44, 'Mencía': 45, 'G-S-M': 46}
    for pos,val in num_to_string.items():
        if x==val:
            x=pos
            return x
    return x

