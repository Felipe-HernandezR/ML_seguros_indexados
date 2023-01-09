# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:45:20 2023

@author: felip
"""

#===================================================================================================================
# Se importan datos y se realiza pre procesamiento
#===================================================================================================================
import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import keras_tuner
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from time import process_time

#%%
#===================================================================================================================
# Se define el modelo y se entrena con Keras_tunner
#===================================================================================================================

class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        # Tune the number of layers.
        num_layers = hp.Int('num_layers', 1, 9)
        for i in range(num_layers):
            with hp.conditional_scope('num_layers', list(range(i + 1, 9 + 1))):
                model.add(
                    layers.Dense(
                        # Tune number of units separately.
                        units= 100,
                        activation=hp.Choice("activation", ["relu", "tanh"]),
                    )
                )
                model.add(
                    layers.Dropout(
                        rate= hp.Choice(f'dropout_{i}', [0.0,0.2,0.4,0.6,0.8])
                    )
                )
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer= "adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(curve="PR")],
        )
        return model

    def fit(self, hp, model, x, y, *args, **kwargs):
        return model.fit(
            x,
            y,
            batch_size = round(np.shape(x)[0]/120),
            
            #batch_size=hp.Choice("batch_size", [32,64, 96, 128]),
            **kwargs,
        )
#%%
t1_start = process_time() 
for i in range(1,2):
    os.chdir(r'C:\Users\felip\OneDrive - Universidad de los andes\Clases Uniandes S-10\Tesis\Datos\Codigo_implementado\UNGRD\Antioquia')
    #Se carga el archivo 
    df = pd.read_csv('dataset_consolidado_antioquia_todosdatos.csv')
    
    X = df.iloc[:,2:13]
    y = df.iloc[:,13]
    
    #Se separa 10% para test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    if i == 1:
        X_train = X_train[['Chirps_runoff_acum','Persiann_runoff_acum',
                           'GPM_runoff_acum','Era5_Surface_runoff',
                           'Era5_volumetric_soil_water_layer_1',
                           'Era5_volumetric_soil_water_layer_2',
                           'Era5_volumetric_soil_water_layer_3',
                           'Era5_volumetric_soil_water_layer_4']]
        X_test = X_test[['Chirps_runoff_acum','Persiann_runoff_acum',
                         'GPM_runoff_acum','Era5_Surface_runoff',
                         'Era5_volumetric_soil_water_layer_1',
                         'Era5_volumetric_soil_water_layer_2',
                         'Era5_volumetric_soil_water_layer_3',
                         'Era5_volumetric_soil_water_layer_4']]
        variables = 8
        nombre = "C_P_G_ER_L1_L2_L3_L4"
    elif i == 2:
        X_train = X_train[['Chirps_runoff_acum','Persiann_runoff_acum',
                           'GPM_runoff_acum','Era5_Surface_runoff',
                           'Era5_volumetric_soil_water_layer_1',
                           'Era5_volumetric_soil_water_layer_2',
                           'Era5_volumetric_soil_water_layer_3']]
        X_test = X_test[['Chirps_runoff_acum','Persiann_runoff_acum',
                         'GPM_runoff_acum','Era5_Surface_runoff',
                         'Era5_volumetric_soil_water_layer_1',
                         'Era5_volumetric_soil_water_layer_2',
                         'Era5_volumetric_soil_water_layer_3']]
        variables = 7
        nombre = "C_P_G_ER_L1_L2_L3"
    elif i == 3:
        X_train = X_train[['Chirps_runoff_acum','Persiann_runoff_acum',
                           'GPM_runoff_acum','Era5_Surface_runoff',
                           'Era5_volumetric_soil_water_layer_1',
                           'Era5_volumetric_soil_water_layer_2']]
        X_test = X_test[['Chirps_runoff_acum','Persiann_runoff_acum',
                         'GPM_runoff_acum','Era5_Surface_runoff',
                         'Era5_volumetric_soil_water_layer_1',
                         'Era5_volumetric_soil_water_layer_2']]
        variables = 6
        nombre = "C_P_G_ER_L1_L2"
    elif i == 4:
        X_train = X_train[['Chirps_runoff_acum','Persiann_runoff_acum',
                           'GPM_runoff_acum','Era5_Surface_runoff',
                           'Era5_volumetric_soil_water_layer_1']]
        X_test = X_test[['Chirps_runoff_acum','Persiann_runoff_acum',
                         'GPM_runoff_acum','Era5_Surface_runoff',
                         'Era5_volumetric_soil_water_layer_1']]
        variables = 5
        nombre = "C_P_G_ER_L1"
    elif i == 5:
        X_train = X_train[['Chirps_runoff_acum','Persiann_runoff_acum',
                           'GPM_runoff_acum','Era5_Surface_runoff']]
        X_test = X_test[['Chirps_runoff_acum','Persiann_runoff_acum',
                         'GPM_runoff_acum','Era5_Surface_runoff']]
        variables = 4
        nombre = "C_P_G_ER"
    elif i == 6:
        X_train = X_train[['Chirps_runoff_acum','Persiann_runoff_acum',
                           'GPM_runoff_acum']]
        X_test = X_test[['Chirps_runoff_acum','Persiann_runoff_acum',
                         'GPM_runoff_acum']]
        variables = 3
        nombre = "C_P_G"
    elif i == 7:
        X_train = X_train[['Chirps_runoff_acum','Persiann_runoff_acum',
                           'Era5_Surface_runoff']]
        X_test = X_test[['Chirps_runoff_acum','Persiann_runoff_acum',
                         'Era5_Surface_runoff']]
        variables = 3
        nombre = "C_P_ER"
    elif i == 8:
        X_train = X_train[['Persiann_runoff_acum','GPM_runoff_acum',
                           'Era5_Surface_runoff']]
        X_test = X_test[['Persiann_runoff_acum','GPM_runoff_acum',
                         'Era5_Surface_runoff']]
        variables = 3
        nombre = "P_G_ER"
    elif i == 9:
        X_train = X_train[['Chirps_runoff_acum','GPM_runoff_acum',
                           'Era5_Surface_runoff']]
        X_test = X_test[['Chirps_runoff_acum','GPM_runoff_acum',
                         'Era5_Surface_runoff']]
        variables = 3
        nombre = "C_G_ER"
    elif i == 10:
        X_train = X_train[['Persiann_runoff_acum','GPM_runoff_acum']]
        X_test = X_test[['Persiann_runoff_acum','GPM_runoff_acum']]
        variables = 2
        nombre = "P_G"
    elif i == 11:
        X_train = X_train[['Chirps_runoff_acum','Persiann_runoff_acum']]
        X_test = X_test[['Chirps_runoff_acum','Persiann_runoff_acum']]
        variables = 2
        nombre = "C_P"
    elif i == 12:
        X_train = X_train[['Chirps_runoff_acum','GPM_runoff_acum']]
        X_test = X_test[['Chirps_runoff_acum','GPM_runoff_acum']]
        variables = 2
        nombre = "C_G"
    elif i == 13:
        X_train = X_train[['Chirps_runoff_acum','Era5_Surface_runoff']]
        X_test = X_test[['Chirps_runoff_acum','Era5_Surface_runoff']]
        variables = 2
        nombre = "C_ER"
    elif i == 14:
        X_train = X_train[['Persiann_runoff_acum','Era5_Surface_runoff']]
        X_test = X_test[['Persiann_runoff_acum','Era5_Surface_runoff']]
        variables = 2
        nombre = "P_ER"
    elif i == 15:
        X_train = X_train[['GPM_runoff_acum','Era5_Surface_runoff']]
        X_test = X_test[['GPM_runoff_acum','Era5_Surface_runoff']]
        variables = 2
        nombre = "G_ER"

    
    #Normalizing the data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #Se separa el 20% para cv
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=5)
    
    os.chdir(r'D:\Documentos\Academico\Uniandes\S-10\Tesis\Resultados')
    
    preprocessing = "under_oversample"
    if preprocessing == "oversampling":
        ros = RandomOverSampler(random_state=40)
        x_t, y_t = ros.fit_resample(X_train, y_train)
    elif preprocessing == "smote":
        sm = SMOTE(random_state=40)
        x_t, y_t = sm.fit_resample(X_train, y_train)
    elif preprocessing == "under_oversample":
        rus = RandomUnderSampler(random_state=40, sampling_strategy= 1/7)
        ros = RandomOverSampler(random_state=40)
        x_t, y_t = rus.fit_resample(X_train, y_train)
        x_t, y_t = ros.fit_resample(x_t, y_t)
    else:
        x_t = X_train
        y_t = y_train
    
    nombre_proyecto = preprocessing + "_tune_hypermodel_" + nombre
    #Se define el algoritmo a utilizar (Hyperband)
    tuner = keras_tuner.Hyperband(
        MyHyperModel(),
        objective=keras_tuner.Objective("val_auc", direction="max"),
        executions_per_trial=2,
        max_epochs=100,
        factor = 4,
        hyperband_iterations= 1,
        seed = 40,
        overwrite=True,
        directory="my_dir_preprocessing_hyperband_auc_todosdatos",
        project_name= nombre_proyecto,
    )
    
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Se utiliza un callback para realizar early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                  mode = 'max', patience=3)
    tuner.search(x_t, y_t, validation_data = (X_test, y_test), verbose = 2, 
                 callbacks = [stop_early]) 

t1_stop = process_time()
print("Elapsed time:", t1_stop, t1_start) 
   
print("Elapsed time during the whole program in seconds:",
                                         t1_stop-t1_start)