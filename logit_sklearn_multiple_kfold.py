# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 22:29:47 2023

@author: felip
"""

#===================================================================================================================
# Se importan datos y se realiza pre procesamiento
#===================================================================================================================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score  
from plotnine import *
import plotnine
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn import metrics
from tensorflow import keras
from sklearn.model_selection import KFold
#%%
#Se grafica el ROC
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
for i in range(4,16):
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
    
# =============================================================================
#     #Vectores para guardar los resultados
#     AUC_test = np.zeros(shape=(4))
#     #AUC_test = np.zeros(shape=(num_modelos,iteraciones))
#     fscore_test = np.zeros(shape=(4))
#     #fscore_test = np.zeros(shape=(num_modelos,iteraciones))
#     binaryCE_test = np.zeros(shape=(4))
#     #binaryCE_test = np.zeros(shape=(num_modelos,iteraciones))
# =============================================================================
    
    num_folds = 5
    
    #Vectores para guardar los resultados
    AUC_cv = np.zeros(shape=(4,num_folds))
    AUC_test = np.zeros(shape=(4))
    fscore_cv = np.zeros(shape=(4,num_folds))
    fscore_test = np.zeros(shape=(4))
    binaryCE_cv = np.zeros(shape=(4,num_folds))
    binaryCE_test = np.zeros(shape=(4))
    
    #Se realiza 5fold CV
    kf = KFold(n_splits=num_folds, random_state= 40, shuffle= True)
    
    y_train = y_train.values.reshape(np.shape(y_train)[0],1)
    y_test = y_test.values.reshape(np.shape(y_test)[0],1)
    
    for fold, (train_index, cv_index) in enumerate(kf.split(X_train, y_train)):
        #print("TRAIN:", train_index, "TEST:", cv_index)
        train_index = train_index.ravel()
        cv_index = cv_index.ravel()
        X_train_fold, X_cv = X_train[train_index], X_train[cv_index]
        y_train_fold, y_cv = y_train[train_index], y_train[cv_index]
        
        for prep in range(0,4):

            if prep == 0:
                ros = RandomOverSampler(random_state=40)
                x_t, y_t = ros.fit_resample(X_train_fold, y_train_fold)
                preprocessing = "oversampling"
            elif prep == 1:
                sm = SMOTE(random_state=40)
                x_t, y_t = sm.fit_resample(X_train_fold, y_train_fold)
                preprocessing = "smote"
            elif prep == 2:
                rus = RandomUnderSampler(random_state=40, sampling_strategy= 1/7)
                ros = RandomOverSampler(random_state=40)
                x_t, y_t = rus.fit_resample(X_train_fold, y_train_fold)
                x_t, y_t = ros.fit_resample(x_t, y_t)
                preprocessing = "under_oversample"
            elif prep == 3:
                x_t = X_train_fold
                y_t = y_train_fold
                preprocessing = "unweighted"
            
            model = LogisticRegression(random_state=0).fit(x_t, y_t)
        
            y_pred_cv = model.predict_proba(X_cv)[:,1]
            # Se compara el mejor threshold del cv con el mejor threshold del test
            
            #Se calcula el binary cross entropy
            bceTensor = keras.losses.binary_crossentropy(y_cv.reshape(y_cv.shape[0],),
                                                         y_pred_cv)
            bce_cv = bceTensor.numpy()
            
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
            index = np.argmax(fscore)
            thresholdOpt = round(thresholds[index], ndigits = 4)
            fscoreOpt_cv = round(fscore[index], ndigits = 4)
            #print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))
            
            # Plot the threshold tuning
            df_threshold_tuning = pd.DataFrame({'F-score':fscore,
                                                'Threshold':thresholds})
            df_threshold_tuning.head()
        
            #lot_precision_recall(y_test, y_pred)
            precision_cv, recall_cv, _ = precision_recall_curve(y_cv, y_pred_cv)
            #print(f'model 1 AUC score: {metrics.auc(recall, precision)}')
            #print(f'model fscore: {fscoreOpt}')
            
            AUC_cv[prep, fold] = metrics.auc(recall_cv, precision_cv)
            fscore_cv[prep, fold] = fscoreOpt_cv
            binaryCE_cv[prep, fold] = bce_cv
            
        
        
    
    for prep in range(0,4):

        if prep == 0:
            ros = RandomOverSampler(random_state=40)
            x_t, y_t = ros.fit_resample(X_train, y_train)
            preprocessing = "oversampling"
        elif prep == 1:
            sm = SMOTE(random_state=40)
            x_t, y_t = sm.fit_resample(X_train, y_train)
            preprocessing = "smote"
        elif prep == 2:
            rus = RandomUnderSampler(random_state=40, sampling_strategy= 1/7)
            ros = RandomOverSampler(random_state=40)
            x_t, y_t = rus.fit_resample(X_train, y_train)
            x_t, y_t = ros.fit_resample(x_t, y_t)
            preprocessing = "under_oversample"
        elif prep == 3:
            x_t = X_train
            y_t = y_train
            preprocessing = "unweighted"
    
        model = LogisticRegression(random_state=0).fit(x_t, y_t)
    
        y_pred = model.predict_proba(X_test)[:,1]
        # Se compara el mejor threshold del cv con el mejor threshold del test
        
        #Se calcula el binary cross entropy
        bceTensor = keras.losses.binary_crossentropy(y_test.reshape(y_test.shape[0]),
                                                     y_pred)
        bce_test = bceTensor.numpy()
        
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
        #print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))
        
        # Plot the threshold tuning
        df_threshold_tuning = pd.DataFrame({'F-score':fscore,
                                            'Threshold':thresholds})
        df_threshold_tuning.head()
    
        #lot_precision_recall(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test,
                                                      y_pred)
        #print(f'model 1 AUC score: {metrics.auc(recall, precision)}')
        #print(f'model fscore: {fscoreOpt}')
        
        AUC_test[prep] = metrics.auc(recall, precision)
        #AUC_test[num, prueba] = metrics.auc(recall, precision)
        fscore_test[prep] = fscoreOpt
        #fscore_test[num, prueba] = fscoreOpt
        binaryCE_test[prep] = bce_test
        #binaryCE_test[num, prueba] = bce_test
    
    AUC_cv_prom = AUC_cv.mean(1)
    #AUC_test_prom = AUC_test.mean(1)
    fscore_cv_prom = fscore_cv.mean(1)
    #fscore_test_prom = fscore_test.mean(1)
    binaryCE_cv_prom = binaryCE_cv.mean(1)
    #binaryCE_test_prom = binaryCE_test.mean(1)
  
    #Se pasan los datos a un excel
    resumen = np.stack([AUC_cv_prom, AUC_test,
                        fscore_cv_prom, fscore_test,
                        binaryCE_cv_prom, binaryCE_test], axis = 1)
    
    df = pd.DataFrame(resumen, columns = ['AUC_cv','AUC_test',
                                          'fscore_cv','fscore_test',
                                          'binaryCE_cv', 'binaryCE_test'],
                      index = ['oversampling','smote',
                              'under_oversample', 'unweighted'])
    
    os.chdir(r'C:\Users\felip\OneDrive - Universidad de los andes\Clases Uniandes S-10\Tesis\Informe\Resultados')
    
    print(nombre)
    with pd.ExcelWriter('resultados_metricas_logit_CV.xlsx',mode='a') as writer:  
        df.to_excel(writer, sheet_name = nombre)
#%%
