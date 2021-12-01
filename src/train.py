## cargo las librerias
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix , r2_score, mean_absolute_error, mean_squared_error 
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
import multiprocessing
from sklearn.svm import SVC
import xgboost
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from scipy.stats import iqr, shapiro, skew 
from scipy import stats, special
import pickle  #para guardar el modelo de ML
import warnings
warnings.filterwarnings('once')

## cargo el archivo con mis funciones creadas para el proyecto
from utils.funciones import *


# rutas
import os
print("Ruta inicial", os.getcwd())
print("Ruta fichero de ejecución", os.path.dirname(__file__))
# cambiar la ruta
os.chdir(os.path.dirname(__file__))
print("Ruta cambiada al fichero de ejecución", os.getcwd())


## cargo el dataset inicial
print("###################################################")
print("###############  EMPIEZA EL PROCESO ###############")
print("###################################################")

df = pd.read_csv("data\\raw\\winemag-data-130k-v2.csv", index_col=0) 

## busco missing values y su ratio
df.isna().sum(axis = 0)
# print("Valores nulos por columna: \n", df.isna().sum(axis = 0))
# print("\nlongitud dataframe inicial: ", len(df.index))
# print("\nratio de missing por columna:\n ", (df.isna().sum(axis = 0))/len(df.index))  

## borro las filas con missing values
df = df.dropna(subset=["country","designation", "price" , "province", "region_1", "taster_name", "variety"])

## borro las columnas que no me aportan ninguna utilidad y encima tienen missing values
df = df.drop(["region_2", "taster_twitter_handle"], axis = 1)

##chequeo missing values y su ratio despues de la primera limpieza
# print("#### Chequeo NaNs después de la primera limpieza ####")
# print("Valores nulos por columna después de la primera limpieza de NaNs: \n", df.isna().sum(axis = 0))
# print("\nratio de missing por columna:\n ", (df.isna().sum(axis = 0))/len(df.index))

## feature engineering columna year
year = []    
for value in df['title']:
    regexresult = re.search(r'19\d{2}|20\d{2}', value)   ## para quedarme solo con el año (valor numérico de 4 dígitos que empieza por 19 o por 20)
    if regexresult:
        year.append(regexresult.group())
    else: year.append(None)

df['year'] = year


# print("Hay {} registros con el año y {} sin él. Estos últimos tendré que borrarlos del dataframe, en cuanto missing values".format(len(df[df['year'].notna()]), len(df[df['year'].isna()].index)))

## ya puedo borrar la columna "title"
df = df.drop(["title"], axis = 1)

## borro los NaNs de la nueva columna "year" y la paso a valor numérico
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

## busco filas duplicadas y las elimino
filas_duplicadas = df[df.duplicated()]
# print("Borrando filas duplicadas... ")
df.drop_duplicates(keep= "first", inplace=True)
# print("Filas borradas!")

## limpio outlier que es un valor erroneo
# print("Enseñando fila con valor muy raro en el precio (2013 USD)...")
err_out = df[df["price"] == 2013]
err_out
# print("Borrando fila...")
df = df.drop(df[df["price"]== 2013].index)
# print("Fila borrada!")

## guardo el dataframe limpio
df.to_csv("data\\processed\\vinos_editado.csv", index=False) 

## REDUCIR DATOS  para que mi recomendador de vino devuelva variedades que sean relativamente fáciles
## de encontrar (variedades de las que se produzcan por lo menos 100 unidades, de las que haya
## por lo menos 20 viñedos y que se produzcan en por lo menos 3 bodegas)

## variedades
frecuencias_variedades = pd.DataFrame(df['variety'].value_counts().sort_values(ascending=False))
mask = frecuencias_variedades["variety"] > 100 
vars_100 = frecuencias_variedades[mask]
# print("Total variedades de las que hay más de 100 unidades:",len(vars_100))
# creo un diccionario con las etiquetas de los vinos y sus respectivos números ordenados según la cantidad de cada vino
char2idx = {u:i for i, u in enumerate(vars_100.index)}
df_variety_num = pd.DataFrame([[key, char2idx[key]] for key in char2idx.keys()], columns=['variety', 'variety_100'])
df_variety_num["variety_100"] = (df_variety_num["variety_100"] + 1) # incremento de uno para que si luego tuviera que escalar esta columna no me de error
vars_100 = df_variety_num.sort_values(by= "variety_100")
df = df.merge(vars_100, how="outer") ## lo mergeo al dataframe general
df = df.dropna(subset=["variety_100"])  ## borro NaNs de esta nueva columna

## viñedos
frec_designation = pd.DataFrame(df['designation'].value_counts().sort_values(ascending=False))
mask = frec_designation["designation"] >= 20  ## defino la máscara
desig_20 = frec_designation[mask]
# print("Total variedades de las que hay más de 20 viñedos:",len(desig_20)) 
desig_20["designation_20"] = desig_20.index
desig_20["designation"] = desig_20.index ## tengo que volver a hacer esto y pasar los nombres a la columna "designation" (ya no me interesa quedarme con las frecuencias
                                         ## y esto es necesario porque luego lo quiero mergear con el dataframe principal)
df = df.merge(desig_20, how="outer")  ## lo mergeo al dataframe general
df = df.dropna(subset=["designation_20"])  ## borro NaNs de esta nueva columna
df.drop(["designation"], axis = 1, inplace=True)  ## ahora la columna "designation" la tengo doble, la borro y me quedo con la que se llama "designation_20"

## bodegas
frec_winery = pd.DataFrame(df['winery'].value_counts().sort_values(ascending=False))
mask = frec_winery["winery"] >= 3  ## defino la máscara
winery_3 = frec_winery[mask]
# print("Total variedades de las que hay más de 3 bodegas:",len(winery_3))
winery_3["winery_3"] = winery_3.index
winery_3["winery"] = winery_3.index  ## tengo que volver a hacer esto y pasar los nombres a la columna "winery" (ya no me interesa quedarme con las frecuencias
                                     ## y esto es necesario porque luego lo quiero mergear con el dataframe principal)
df = df.merge(winery_3, how="outer")   ## lo mergeo al dataframe general
df = df.dropna(subset=["winery_3"])   ## borro NaNs de esta nueva columna
df.drop(["winery"], axis = 1, inplace=True)  ## ahora la columna "winery" la tengo doble, la borro y me quedo con la que se llama "winery_3"

## guardo una version de csv CON LA COLUMNA "variety" original, me servirá luego si uso el NLP y tokenizer
# print("Guardando csv para futuro modelo de NLP")
df.to_csv("data\\processed\\vinos_para_NLP_model.csv", index=False) 
# print("Borrando columna 'variety' original y guardando csv para modelos de ML, donde tengo la columna 'variety_100', numérica...")
df.drop(["variety"], axis = 1, inplace=True)  
df.to_csv("data\\processed\\vinos_var100_des20_win3.csv", index=False) 
# print("Ya se ha guardado el dataframe para el modelo de Machine Learning!\nEsta versión tiene outliers.")



## cargo el dataframe con  outliers para empezar con el modelo de ML
## antes borro las columnas que no me sirven, luego hago el get dummies y luego procedo a borrar los outliers
df_4 = pd.read_csv("data\\processed\\vinos_var100_des20_win3.csv")
df_4.drop(["description","province", "taster_name", "country"], axis = 1, inplace= True)
df_4_dumm = pd.get_dummies(data = df_4)
df_4_dumm_no_out = outliers_quantie(df_4_dumm, 'price')  ## dejo el parámetro por defect de 1.5, así veo que me ha quitado muchos más outliers que en las pruebas anteriores
# print("Largo original del dataframe:", len(df_4_dumm))
# print("Largo del dataframe sin outliers en price:", len(df_4_dumm_no_out))

## defino "X"e "y"
X4 = df_4_dumm_no_out.drop(["variety_100"], axis = 1)
y4 = np.array(df_4_dumm_no_out["variety_100"]).reshape(-1,)

## separo en train y test
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.2, random_state=12)

## pipeline
reg_log4 = Pipeline(steps = [
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("reglog4", LogisticRegression(C=21.544346900318832, max_iter=1000, penalty="l2"))
])

## entreno
reg_log4.fit(X4_train, y4_train)


## best score en TEST
predictions = reg_log4.predict(X4_test)
accuracy = accuracy_score(y4_test, predictions)



##guardo las predicciones vs valores reales en un dataframe
df_preds = pd.DataFrame(predictions, columns = ["PREDICCIONES"])
df_preds["VALOR REAL"] = pd.Series(y4_test)  ## hasta aquí solo salen números
## aplico funcion para reconvertirlos a nombres
df_preds["VALOR REAL nombre"] = df_preds["VALOR REAL"].apply(decodificar_valores)
df_preds["PREDICCIONES nombre"] = df_preds["PREDICCIONES"].apply(decodificar_valores)
## solo me interesa ver el nombre del vino, si el modelo lo ha adivinado o no, no me interesan los números
df_preds_variety = df_preds[["VALOR REAL nombre", "PREDICCIONES nombre"]]
## guardo el dataframe con las predicciones
df_preds_variety.to_csv("data\\processed\\predicciones_vs_reales.csv", index=False)

## guardo el modelo
filename = 'models/Modelo_Reg_Log.model' 

with open(filename, 'wb') as archivo_salida:
    pickle.dump(reg_log4, archivo_salida)


print("\n\nEstas son las {} predicciones: {}".format(len(predictions),predictions))
print("Con los datos de TEST he obtenido un score de: ",reg_log4.score(X4_test, y4_test))
print("Redondeando, tengo una precisión del {}%".format(round(accuracy*100)) )
print("##################################################################")
print("###################  EL PROCESO HA FINALIZADO ####################")
print("##################################################################")
