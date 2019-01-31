# Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
import numpy as np

def save_to_bin_file(mlpClassifierObj, fpath_save):
    
    map_actFunc_to_num = {
        'identity': 0,
        'logistic': 1,
        'tanh': 2,
        'relu': 3,
        'softmax': 4
    }

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

    W_layers = mlpClassifierObj.coefs_
    B_layers = mlpClassifierObj.intercepts_
    W_hidden = W_layers[0]
    B_hidden = B_layers[0].reshape(1, -1)
    W_output = W_layers[1]
    B_output = B_layers[1].reshape(1, -1)
    activation_hidden = map_actFunc_to_num[mlpClassifierObj.activation]
    activation_output = map_actFunc_to_num[mlpClassifierObj.out_activation_]

    assert len(W_layers) == 2, 'There must be only be one hidden layer, otherwise cannot use this function to extract weights'

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