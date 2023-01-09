# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:15:49 2022

@author: felip
"""
#===================================================================================================================
# Se importan datos y se realiza pre procesamiento
#===================================================================================================================
import pandas as pd
import numpy as np
import os
 
# Function to Get the current
# working directory
def current_path():
    print("Current working directory before")
    print(os.getcwd())
    print()
 
 
# Driver's code
# Printing CWD before
current_path()

os.chdir(r'C:\Users\felip\OneDrive - Universidad de los andes\Clases Uniandes S-10\Tesis\Datos\Codigo_implementado\UNGRD\Antioquia')

current_path()

#Se carga el archivo 
df = pd.read_csv('dataset_consolidado_antioquia.csv')

df.head(10)

X = df.iloc[:,2:13]
y = df.iloc[:,13]

#Se separa 10% para test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

list(X_train.columns)[0:15]
#Definen las variables de entrada al modelo
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

#Normalizacion
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Frecuencia en el y_train
print(y_train.value_counts())

#Frecuencia en el y_test
print(y_test.value_counts())

#Se separa el 20% para cv
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=5)
#Frecuencia en el y_cv
print(y_cv.value_counts())
#%%
#===================================================================================================================
# Se define el modelo y se llama el modelo guardado
#===================================================================================================================
os.chdir(r'D:\Documentos\Academico\Uniandes\S-10\Tesis\Resultados')
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import keras_tuner
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
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
from tensorflow import keras
import tensorflow as tf
import keras_tuner

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
    project_name="oversampling_tune_hypermodel_C_P_G_ER_L1_L2_L3_L4",
)


tuner.reload()
tuner.get_best_models(num_models=10)
#%%
tuner.search_space_summary()

# Get the top 10 models.
models = tuner.get_best_models(num_models=60)
num = 57
best_model = models[num]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(None, 8))
best_model.summary()

tuner.results_summary()

best_model.summary()

#%%
#===================================================================================================================
# Se halla el mejor modelo y se calibra el threshold con el cv set
#===================================================================================================================
#Se entrena el mejor modelo sin el cv para realizar tunning del threshold
#Se construye el mejor modelo
# Get the top 10 models.
preprocessing = "oversampling"

#models = tuner.get_best_models(num_models=num_modelos)
best_hps=tuner.get_best_hyperparameters(num_trials=60)

params = best_hps[num]
print("==================================================================")
print(f"Corriendo modelo {num}")
print("Parametros:")
print(f'Numero de capas: {params.get("num_layers")}')
print(f'Funcion de activacion: {params.get("activation")}')
for capa in range(0,params.get("num_layers")):   
    print(f'Regularización: {params.get("dropout_" + str(capa))}')

#best_model = models[num]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
#best_model.build(input_shape=(None, variables))
best_model = tuner.hypermodel.build(params)
# Fit with the entire dataset.
#preprocessing = "oversampling"
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
    x_t, y_t = ros.fit_resample(X_train, y_train)
else:
    x_t = X_train
    y_t = y_train
    
bz = round(np.shape(x_t)[0]/120)

# Se utiliza un callback para realizar early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = best_model.fit(x_t, y_t, batch_size = bz,validation_data = (X_cv,y_cv), epochs=100,
                         callbacks = [stop_early])

y_pred_cv = best_model.predict(X_cv)

bceTensor = keras.losses.binary_crossentropy(y_cv, y_pred_cv.reshape(y_pred_cv.shape[0],))
bce_cv = bceTensor.numpy()
# Se halla el mejor threshold_cv
from sklearn.metrics import f1_score  
# Array for finding the optimal threshold
thresholds = np.arange(0.0, 1.0, 0.0001)
fscore = np.zeros(shape=(len(thresholds)))
print('Length of sequence: {}'.format(len(thresholds)))

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

#===================================================================================================================
# Se utilizan los parametros calibrados para determinar el desempeño en el test set
#===================================================================================================================
#Se entrena el mejor modelo con todos los datos
# Get the top 10 models.
#models = tuner.get_best_models(num_models=num_modelos)
#best_model = models[num]
best_model = models[num]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(None, 8))

params = best_hps[num]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
#best_model.build(input_shape=(None, variables))
best_model = tuner.hypermodel.build(params)

#Se junta el train y cv
y_t = y_train.values.reshape(np.shape(y_train)[0],1)
y_cval = y_cv.values.reshape(np.shape(y_cv)[0],1)

#Se junta el train y el cv
x_t = np.vstack((X_train, X_cv))
y_t = np.vstack((y_t, y_cval))

#Se realiza el re-sampling
if preprocessing == "oversampling":
    ros = RandomOverSampler(random_state=40)
    x_t, y_t = ros.fit_resample(x_t, y_t)
elif preprocessing == "smote":
    sm = SMOTE(random_state=40)
    x_t, y_t = sm.fit_resample(x_t, y_t)
elif preprocessing == "under_oversample":
    rus = RandomUnderSampler(random_state=40, sampling_strategy= 1/7)
    ros = RandomOverSampler(random_state=40)
    x_t, y_t = rus.fit_resample(x_t, y_t)
    x_t, y_t = ros.fit_resample(x_t, y_t)
else:
    x_t = x_t
    y_t = y_t

#bz = round(np.shape(x_t)[0]/120)
tuner.reload()
# Fit with the entire dataset.
#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = best_model.fit(x_t, y_t,batch_size = bz, validation_data = (X_test,y_test), epochs=100,
                         callbacks = [stop_early])


y_pred = best_model.predict(X_test)

bceTensor = keras.losses.binary_crossentropy(y_test, y_pred.reshape(y_pred.shape[0],))
bce_test = bceTensor.numpy()
#%%
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    if (y_pred[i] > thresholdOpt_cv):
        pred.append(1)
    else:
        pred.append(0)
    

print(y_pred[np.argmin(y_pred)])

my_series = pd.Series(pred)
print(my_series.value_counts()) #Ninugn 1

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,y_test)
print('Accuracy is:', a*100)


#Tabla de predicciones
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))

#%%
# Se compara el mejor threshold del cv con el mejor threshold del test
from sklearn.metrics import f1_score  
from plotnine import *
import plotnine
# Array for finding the optimal threshold
thresholds = np.arange(0.0, 1.0, 0.0001)
fscore = np.zeros(shape=(len(thresholds)))
print('Length of sequence: {}'.format(len(thresholds)))

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
print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))

# Plot the threshold tuning
df_threshold_tuning = pd.DataFrame({'F-score':fscore,
                                    'Threshold':thresholds})
df_threshold_tuning.head()

#Se grafica el ROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn import metrics

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
    

plot_roc_curve(y_test, y_pred)
plot_roc_curve(y_cv, y_pred_cv)
print(f'model 1 AUC score: {roc_auc_score(y_test, y_pred)}')

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

plot_precision_recall(y_test, y_pred)
plot_precision_recall(y_cv, y_pred_cv)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
precision_cv, recall_cv, _ = precision_recall_curve(y_cv, y_pred_cv)

print(f'model cv AUC score: {metrics.auc(recall_cv, precision_cv)}')
print(f'model test AUC score: {metrics.auc(recall, precision)}')
print(f'model fscore_cv: {fscoreOpt_cv}')
print(f'model fscore_test: {fscoreOpt}')
print(f'model bce_cv: {bce_cv}')
print(f'model bce_test: {bce_test}')

#%%
plotnine.options.figure_size = (8, 4.8)
#%%
(
    ggplot(data = df_threshold_tuning)+
    geom_point(aes(x = 'Threshold',
                   y = 'F-score'),
               size = 0.4)+
    # Best threshold
    geom_point(aes(x = thresholdOpt,
                   y = fscoreOpt),
               color = '#981220',
               size = 4)+
    # Best threshold cv
    geom_point(aes(x = thresholdOpt_cv,
                   y = fscore[index_cv]),
               color = '#000080',
               size = 4)+
    geom_line(aes(x = 'Threshold',
                   y = 'F-score'))+
    # Annotate the text
    geom_text(aes(x = thresholdOpt,
                  y = fscoreOpt),
              label = 'Optimal threshold \n for class: {}'.format(thresholdOpt),
              nudge_x = 0.8,
              nudge_y = 0,
              size = 10,
              fontstyle = 'italic')+
    # Annotate the text
    geom_text(aes(x = thresholdOpt_cv,
                  y = fscoreOpt_cv),
              label = 'Optimal CV threshold \n for class: {}'.format(thresholdOpt_cv),
              nudge_x = 0.8,
              nudge_y = -.05,
              size = 10,
              fontstyle = 'italic')+
    labs(title = 'Threshold Tuning Curve')+
    xlab('Threshold')+
    ylab('F-score')+
    theme_gray()
)
#%%
#%%
import matplotlib.pyplot as plt
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

