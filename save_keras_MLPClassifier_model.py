# Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
import numpy as np

def save_to_bin_file(kerasClassifierModel, fpath_save):

    map_actFuncKeras_to_actFunc = {
        'softmax': 'softmax',
        'relu': 'relu',
        'tanh': 'tanh',
        'sigmoid': 'logistic',
        'linear': 'identity'
    }
    
    map_actFunc_to_num = {
        'identity': 0,
        'logistic': 1,
        'tanh': 2,
        'relu': 3,
        'softmax': 4
    }

    hidLayerAct_keras = kerasClassifierModel.layers[0].get_config()['activation']
    outputLayerAct_keras = kerasClassifierModel.layers[-1].get_config()['activation']

    assert hidLayerAct_keras in map_actFuncKeras_to_actFunc, 'The trained Keras model must have hidden layer activation function to be one of: ' + ','.join(map_actFuncKeras_to_actFunc.keys())
    assert outputLayerAct_keras is 'softmax', 'The trained keras model must have softmax as the output layer activation function'

    hidLayerAct = map_actFuncKeras_to_actFunc[hidLayerAct_keras]

    assert not np.any([len(kerasClassifierModel.layers[i].get_weights()) for i in range(1, len(kerasClassifierModel.layers) - 1)]), 'The Keras model must have only one hidden layer.'

    # Let input feature vector be x
    # Let number of input neurons be M
    # Let number of hidden neurons be H
    # Let number of output neurons be O
    # x.shape = (1, M)
    # W_hidden.shape = (M, H)
    # B_hidden.shape = (H,)
    # w_output.shape = (H, O)
    # b_output.shape = (O,)
    # to project input feature vector to hidden space:
    # x_proj = x * W_hidden ==> x_proj.shape = (1,H)
    # x_proj = actFunc(x_proj + B_hidden) ==> x_proj_shape = (1,H)
    # to project from hidden space to output space
    # y_pred = x_proj * w_output ==> y_pred.shape = (1,O)
    # y_pred = actFunc(y_pred + b_output) ==>  y_pred.shape = (1,O)

    [W_hidden, B_hidden] = kerasClassifierModel.layers[0].get_weights()
    [W_output, B_output] = kerasClassifierModel.layers[-1].get_weights()
    B_hidden = B_hidden.reshape(1, -1).astype(np.float64)
    B_output = B_output.reshape(1, -1).astype(np.float64)

    W_hidden = W_hidden.astype(np.float64)
    B_output = B_output.astype(np.float64)

    activation_hidden = map_actFunc_to_num[hidLayerAct]
    activation_output = map_actFunc_to_num[outputLayerAct_keras]

    # print(W_hidden.shape)
    # print(B_hidden.shape)
    # print(W_output.shape)
    # print(B_output.shape)

    weights_vec_mlp = np.r_[W_hidden.shape[0], W_hidden.shape[1], W_hidden.flatten(),
                            B_hidden.shape[0], B_hidden.shape[1], B_hidden.flatten(),
                            W_output.shape[0], W_output.shape[1], W_output.flatten(),
                            B_output.shape[0], B_output.shape[1], B_output.flatten(),
                            activation_hidden, activation_output]

    with open(fpath_save, 'wb') as fid:
        weights_vec_mlp.tofile(fid)