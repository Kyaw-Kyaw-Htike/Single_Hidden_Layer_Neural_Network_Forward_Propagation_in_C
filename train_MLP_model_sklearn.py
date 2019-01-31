# Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
import sklearn.datasets
import sklearn.neural_network
import numpy as np
from sklearn.model_selection import train_test_split
import utils_cur
import save_sklearn_MLPClassifier_model

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

# idx_rem = np.where(y==2)
# X = np.delete(X, idx_rem, axis=0)
# y = np.delete(y, idx_rem, axis=0)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)

params_mlp = {'hidden_layer_sizes': (5,),
              'activation': 'logistic',
              'solver': 'lbfgs',
              'max_iter': 1000,
              'early_stopping':True,
              'validation_fraction':0.3
              }
mlpobj = sklearn.neural_network.MLPClassifier(**params_mlp).fit(X_train,y_train)
utils_cur.eval_classifier(mlpobj, X_test, y_test)
save_sklearn_MLPClassifier_model.save_to_bin_file(mlpobj, 'model.bin')

utils_cur.predict_test_case(mlpobj, 12, X_test, y_test, True)

import pickle

save_sklearn_MLPClassifier_model.save_to_bin_file(pickle.load(open(r"D:\Working_on_currently\\Biometric_classification\AgeGenderEthnicity\UsingDlib\biometric_classify_model_oldData_1stProto_DlibFeat_sklearnMLP_ethnicity.pk", 'rb')), r"D:\Working_on_currently\Biometric_classification\AgeGenderEthnicity\UsingDlib\biometric_classify_model_oldData_1stProto_DlibFeat_sklearnMLP_ethnicity.bin")


