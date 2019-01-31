# Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
import sklearn.datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.preprocessing import OneHotEncoder
import keras.callbacks
import numpy as np
import keras.optimizers
import keras.regularizers
import sklearn.neural_network
import utils_cur
import save_keras_MLPClassifier_model

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)

onehot_encoder = OneHotEncoder(sparse=False)
y_train_hot = onehot_encoder.fit_transform(y_train.reshape(-1,1))
y_test_hot = onehot_encoder.fit_transform(y_test.reshape(-1,1))

model = Sequential()
model.add(Dense(units=40, activation='relu', input_dim=4))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
model.fit(X_train, y_train_hot, callbacks=[earlyStopping], validation_split=0.3, shuffle=True, epochs=1000)
loss_and_metrics = model.evaluate(X_test, y_test_hot)
loss_and_metrics_train = model.evaluate(X_train, y_train_hot)
print(loss_and_metrics)
print(loss_and_metrics_train)
probs_pred_train = model.predict(X_train)
probs_pred_test = model.predict(X_test)
labels_pred_train = np.argmax(probs_pred_train, axis=1)
labels_pred_test = np.argmax(probs_pred_test, axis=1)

len(model.layers)

import imp
imp.reload(save_keras_MLPClassifier_model)
save_keras_MLPClassifier_model.save_to_bin_file(model, 'model_keras.bin')

save_keras_MLPClassifier_model.save_to_bin_file(keras.models.load_model(r"D:\Working_on_currently\Biometric_classification\AgeGenderEthnicity\UsingDlib\trained_model_age_classification_keras_mlp.h5"), r"D:\Working_on_currently\Biometric_classification\AgeGenderEthnicity\UsingDlib\trained_model_age_classification_keras_mlp.bin")

aa = np.any([len(model.layers[i].get_weights()) for i in range(1, len(model.layers)-1) ])
aa = model.layers[1].get_weights()[0]

[W_hidden, B_hidden] = model.layers[0].get_weights()
[W_output, B_output] = model.layers[2].get_weights()

[W_hidden, B_hidden] = model.layers[0].get_weights()
[W_output, B_output] = model.layers[-1].get_weights()

model.save('keras_model.h5')

i = 3
xcur = X_test[i].reshape(1,-1)
for i, cc in enumerate(xcur[0]):
    print('input_fvec[{}] = {};'.format(i, cc))
probs_pred_test = model.predict(xcur)[0]
print('func: {}, {}, {}'.format(probs_pred_test[0], probs_pred_test[1], probs_pred_test[2]))

# manual
x_proj = np.maximum(0, np.matmul(xcur, W_hidden) + B_hidden)
probs_pred_test = utils_cur.softmax_v1(np.matmul(x_proj, W_output) + B_output)[0]
print('manu: {}, {}, {}'.format(probs_pred_test[0], probs_pred_test[1], probs_pred_test[2]))

# using prediction function
probs_pred_test = model.predict(xcur)[0]
print('func: {}, {}, {}'.format(probs_pred_test[0], probs_pred_test[1], probs_pred_test[2]))

np.max(aa, axis=1)

params_mlp = {'hidden_layer_sizes': (40,),
              'activation': 'relu',
              'solver': 'lbfgs',
              'max_iter': 1000,
              'early_stopping':True,
              'validation_fraction':0.3
              }
mlpobj = sklearn.neural_network.MLPClassifier(**params_mlp).fit(X_train,y_train)
print('Training score=', mlpobj.score(X_train, y_train))
print('Test score=', mlpobj.score(X_test, y_test))

