// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once
#include <cstdio>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixMLP;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> RowVecMLP;

class MLPClassifier_sklearn
{
private:

	enum Activation { IDENTITY, LOGISTIC, RELU, TANH, SOFTMAX };
	Activation map_actFunc_num_to_enum[5] = { IDENTITY, LOGISTIC, TANH, RELU, SOFTMAX }; // must correspond to the one in my Python code
	MatrixMLP W_hidden;
	MatrixMLP B_hidden;
	MatrixMLP W_output;
	MatrixMLP B_output;
	Activation actFuncHidden;
	Activation actFuncOutput;

	MatrixMLP read_matrix(FILE* fid);
	RowVecMLP apply_actFunc(Activation activation, RowVecMLP v);

public:

	MLPClassifier_sklearn(std::string fpath_model);
	RowVecMLP project_onto_hidden_layer(RowVecMLP input_feature_vector);
	RowVecMLP project_onto_output_layer(RowVecMLP input_feature_vector);
	int predict_class_label(RowVecMLP input_feature_vector);
	RowVecMLP predict_class_prob(RowVecMLP input_feature_vector);

};