# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 20:02:31 2023

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
from sklearn.model_selection import KFold
from time import process_time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn import metrics
#%%
#Funcion para hallar precision, recall y graficar el roc
def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    plt.show()
    

#plot_roc_curve(y_test, y_pred)
#print(f'model 1 AUC score: {roc_auc_score(y_test, y_pred)}')

def plot_precision_recall(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    precision, recall, _ = precision_recall_curve(true_y, y_prob)
    no_skill = len(true_y[true_y==1]) / len(true_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color = 'r', label='No Skill')
    auc = metrics.auc(recall, precision)
    plt.plot(recall, precision, label='PR curve (area = %.2f)' %auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend()
    plt.show()
#%%
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
    df = pd.read_csv('dataset_consolidado_antioquia.csv')
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
    
    #Normalizing the data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #===================================================================================================================
    # Se define el modelo y se llama el modelo guardado
    #===================================================================================================================
    
    y_train = y_train.values.reshape(np.shape(y_train)[0],1)
    y_test = y_test.values.reshape(np.shape(y_test)[0],1)
    
    for prep in range(1,2):
        
        os.chdir(r'D:\Documentos\Academico\Uniandes\S-10\Tesis\Resultados')
        
        if prep == 0:
            preprocessing = "oversampling"
        elif prep == 1:
            preprocessing = "smote"
        elif prep == 2:
            preprocessing = "under_oversample"
        elif prep == 3:
            preprocessing = "unweighted"
        
        nombre_proyecto = preprocessing + "_tune_hypermodel_" + nombre
        
        tuner = keras_tuner.Hyperband(
            MyHyperModel(),
            objective=keras_tuner.Objective("val_loss", direction="min"),
            executions_per_trial=2,
            max_epochs=100,
            factor = 4,
            hyperband_iterations= 1,
            seed = 40,
            overwrite=False,
            directory="my_dir_preprocessing_hyperband_loss_todosdatos",
            project_name= nombre_proyecto,
        )
        
        tuner.reload()    
        
        models = tuner.get_best_models(num_models=10)
        num = 0
        model = models[num]
        # Build the model.
        # Needed for `Sequential` without specified `input_shape`.
        model.build(input_shape=(None, variables))
        
        #Se define el loop, 50 mejores modelos, 5 folds
        num_modelos = 20
        iteraciones = 3
        num_folds = 5
        
        #Vectores para guardar los resultados
        AUC_cv = np.zeros(shape=(num_modelos,num_folds))
        AUC_test = np.zeros(shape=(num_modelos,iteraciones))
        fscore_cv = np.zeros(shape=(num_modelos,num_folds))
        fscore_test = np.zeros(shape=(num_modelos,iteraciones))
        binaryCE_cv = np.zeros(shape=(num_modelos,num_folds))
        binaryCE_test = np.zeros(shape=(num_modelos,iteraciones))
        capas = np.zeros(shape = (num_modelos))
        funcion = ["" for x in range(num_modelos)]
        
        #Se realiza 5fold CV
        kf = KFold(n_splits=num_folds, random_state= 40, shuffle= True)
        
        best_hps=tuner.get_best_hyperparameters(num_trials=num_modelos)

        for num in range(0,num_modelos):
            params = best_hps[num]
            print("==================================================================")
            print(f"Corriendo modelo {num}")
            print("Parametros:")
            print(f'Numero de capas: {params.get("num_layers")}')
            capas[num] = params.get("num_layers")
            print(f'Funcion de activacion: {params.get("activation")}')
            funcion[num] = str(params.get("activation"))
            for capa in range(0,params.get("num_layers")):   
                print(f'Regularización: {params.get("dropout_" + str(capa))}')

            #best_model = models[num]
            # Build the model.
            # Needed for `Sequential` without specified `input_shape`.
            #best_model.build(input_shape=(None, variables))
            # Fit with the entire dataset.
            
            for fold, (train_index, cv_index) in enumerate(kf.split(X_train, y_train)):
                
                
                best_model = tuner.hypermodel.build(params)
                best_model.build(input_shape=(None, variables))
                #print("TRAIN:", train_index, "TEST:", cv_index)
                train_index = train_index.ravel()
                cv_index = cv_index.ravel()
                X_train_fold, X_cv = X_train[train_index], X_train[cv_index]
                y_train_fold, y_cv = y_train[train_index], y_train[cv_index]
        
                if preprocessing == "oversampling":
                    ros = RandomOverSampler(random_state=40)
                    x_t, y_t = ros.fit_resample(X_train_fold, y_train_fold)
                elif preprocessing == "smote":
                    sm = SMOTE(random_state=40)
                    x_t, y_t = sm.fit_resample(X_train_fold, y_train_fold)
                elif preprocessing == "under_oversample":
                    rus = RandomUnderSampler(random_state=40, sampling_strategy= 1/7)
                    ros = RandomOverSampler(random_state=40)
                    x_t, y_t = rus.fit_resample(X_train_fold, y_train_fold)
                    x_t, y_t = ros.fit_resample(x_t, y_t)
                else:
                    x_t = X_train_fold
                    y_t = y_train_fold
                    
                bz = round(np.shape(x_t)[0]/120)
            
                # Se utiliza un callback para realizar early stopping
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                history = best_model.fit(x_t, y_t, batch_size = bz,validation_data = (X_cv,y_cv), epochs=100,
                                         callbacks = [stop_early])
                
                y_pred_cv = best_model.predict(X_cv)
                
                bceTensor = keras.losses.binary_crossentropy(y_cv.reshape(y_cv.shape[0],),
                                                             y_pred_cv.reshape(y_pred_cv.shape[0],))
                bce_cv = bceTensor.numpy()
                # Se halla el mejor threshold_cv
                from sklearn.metrics import f1_score  
                # Array for finding the optimal threshold
                thresholds = np.arange(0.0, 1.0, 0.0001)
                fscore = np.zeros(shape=(len(thresholds)))
                #print('Length of sequence: {}'.format(len(thresholds)))
                
                # Fit the model
                for index, elem in enumerate(thresholds):
                    # Corrected probabilities
                    y_pred_prob = (y_pred_cv > elem).astype('int')
                    # Calculate the f-score
                    fscore[index] = f1_score(y_cv, y_pred_prob)
                
                # Find the optimal threshold
                index_cv = np.argmax(fscore)
                thresholdOpt_cv = round(thresholds[index_cv], ndigits = 4)
                fscoreOpt_cv = round(fscore[index_cv], ndigits = 4)
                #print(thresholds[index_cv])
                
                #plot_precision_recall(y_test, y_pred)
                #precision, recall, _ = precision_recall_curve(y_test, y_pred)
                precision_cv, recall_cv, _ = precision_recall_curve(y_cv, y_pred_cv)
                
                print(f'model cv AUC score: {metrics.auc(recall_cv, precision_cv)}')
                #print(f'model test AUC score: {metrics.auc(recall, precision)}')
                print(f'model fscore_cv: {fscoreOpt_cv}')
                #print(f'model fscore_test: {fscoreOpt}')
                print(f'model bce_cv: {bce_cv}')
                #print(f'model bce_test: {bce_test}')
                
                
                AUC_cv[num, fold] = metrics.auc(recall_cv, precision_cv)
                #AUC_test[num, prueba] = metrics.auc(recall, precision)
                fscore_cv[num, fold] = fscoreOpt_cv
                #fscore_test[num, prueba] = fscoreOpt
                binaryCE_cv[num, fold] = bce_cv
                #binaryCE_test[num, prueba] = bce_test
                
            if preprocessing == "oversampling":
                ros = RandomOverSampler(random_state=40)
                x_t, y_t = ros.fit_resample(X_train, y_train)
                bz = 88
            elif preprocessing == "smote":
                sm = SMOTE(random_state=40)
                x_t, y_t = sm.fit_resample(X_train, y_train)
                bz = 88
            elif preprocessing == "under_oversample":
                rus = RandomUnderSampler(random_state=40, sampling_strategy= 1/7)
                ros = RandomOverSampler(random_state=40)
                x_t, y_t = rus.fit_resample(X_train, y_train)
                x_t, y_t = ros.fit_resample(x_t, y_t)
                bz = 42
            else:
                x_t = X_train
                y_t = y_train
                bz = 47
            
            for prueba in range(0,iteraciones):
                #best_model = models[num]
                # Build the model.
                # Needed for `Sequential` without specified `input_shape`.
                #best_model.build(input_shape=(None, variables))
                best_model = tuner.hypermodel.build(params)
                best_model.build(input_shape=(None, variables))
                # Fit with the entire dataset.
                
                # Se utiliza un callback para realizar early stopping
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                history = best_model.fit(x_t, y_t, batch_size = bz,validation_data = (X_test,y_test), epochs=100,
                                         callbacks = [stop_early])
                
                y_pred = best_model.predict(X_test)
                
                bceTensor = keras.losses.binary_crossentropy(y_test.reshape(y_test.shape[0],),
                                                             y_pred.reshape(y_pred.shape[0],))
                bce_test = bceTensor.numpy()
                # Se halla el mejor threshold_cv
                from sklearn.metrics import f1_score  
                # Array for finding the optimal threshold
                thresholds = np.arange(0.0, 1.0, 0.0001)
                fscore = np.zeros(shape=(len(thresholds)))
                #print('Length of sequence: {}'.format(len(thresholds)))
                
                # Fit the model
                for index, elem in enumerate(thresholds):
                    # Corrected probabilities
                    y_pred_prob = (y_pred > elem).astype('int')
                    # Calculate the f-score
                    fscore[index] = f1_score(y_test, y_pred_prob)
                
                # Find the optimal threshold
                index = np.argmax(fscore)
                thresholdOpt = round(thresholds[index], ndigits = 4)
                fscoreOpt = round(fscore[index], ndigits = 4)
                #print(thresholds[index_cv])
                
                #plot_precision_recall(y_test, y_pred)
                #precision, recall, _ = precision_recall_curve(y_test, y_pred)
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                
                #print(f'model cv AUC score: {metrics.auc(recall_cv, precision_cv)}')
                print(f'model test AUC score: {metrics.auc(recall, precision)}')
                #print(f'model fscore_cv: {fscoreOpt_cv}')
                print(f'model fscore_test: {fscoreOpt}')
                #print(f'model bce_cv: {bce_cv}')
                print(f'model bce_test: {bce_test}')
                
                
                #AUC_cv[num, prueba] = metrics.auc(recall_cv, precision_cv)
                AUC_test[num, prueba] = metrics.auc(recall, precision)
                #fscore_cv[num, prueba] = fscoreOpt_cv
                fscore_test[num, prueba] = fscoreOpt
                #binaryCE_cv[num, prueba] = bce_cv
                binaryCE_test[num, prueba] = bce_test
                
            
        AUC_cv_prom = AUC_cv.mean(1)
        AUC_test_prom = AUC_test.mean(1)
        fscore_cv_prom = fscore_cv.mean(1)
        fscore_test_prom = fscore_test.mean(1)
        binaryCE_cv_prom = binaryCE_cv.mean(1)
        binaryCE_test_prom = binaryCE_test.mean(1)
        
        
        #Se pasan los datos a un excel
        resumen = np.stack([AUC_cv_prom, AUC_test_prom,
                            fscore_cv_prom, fscore_test_prom,
                            binaryCE_cv_prom, binaryCE_test_prom,
                            capas, funcion], axis = 1)
                                              
        df = pd.DataFrame(resumen, columns = ['AUC_cv','AUC_test',
                                              'fscore_cv','fscore_test',
                                              'binaryCE_cv', 'binaryCE_test',
                                              'num_capas', 'funcion'])
        
        os.chdir(r'C:\Users\felip\OneDrive - Universidad de los andes\Clases Uniandes S-10\Tesis\Informe\Resultados')
        
        nombre_hoja = preprocessing + '_8_retrained'
        with pd.ExcelWriter('resultados_metricas_test_top200.xlsx',mode='a') as writer:  
            df.to_excel(writer, sheet_name = nombre_hoja)
    
    t1_stop = process_time()
    print("Elapsed time:", t1_stop, t1_start) 
       
    print("Elapsed time during the whole program in seconds:",
                                             t1_stop-t1_start)
