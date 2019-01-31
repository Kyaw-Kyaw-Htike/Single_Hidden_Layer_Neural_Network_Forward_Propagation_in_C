// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "MLPClassifier_sklearn.h"
#include <vector>

MatrixMLP MLPClassifier_sklearn::read_matrix(FILE * fid)
{
	std::vector<double> array_shape(2);
	fread(array_shape.data(), sizeof(double), 2, fid);
	int nrows = array_shape[0];
	int ncols = array_shape[1];
	MatrixMLP m(nrows, ncols);
	fread(m.data(), sizeof(double), nrows * ncols, fid);
	return m;
}

RowVecMLP MLPClassifier_sklearn::apply_actFunc(Activation activation, RowVecMLP v) {
	switch (activation) {
	case LOGISTIC:
		for (int i = 0, l = v.size(); i < l; i++) {
			v[i] = 1. / (1. + exp(-v[i]));
		}
		break;
	case RELU:
		for (int i = 0, l = v.size(); i < l; i++) {
			v[i] = std::max(0.0, v[i]);
		}
		break;
	case TANH:
		for (int i = 0, l = v.size(); i < l; i++) {
			v[i] = std::tanh(v[i]);
		}
		break;
	case SOFTMAX:
		double max = std::numeric_limits<double>::lowest();
		for (size_t i = 0; i < v.size(); i++)
		{
			if (v[i] > max) {
				max = v[i];
			}
		}
		for (int i = 0, l = v.size(); i < l; i++) {
			v[i] = exp(v[i] - max);
		}
		double sum = 0.;
		for (size_t i = 0; i < v.size(); i++)
		{
			sum += v[i];
		}
		for (int i = 0, l = v.size(); i < l; i++) {
			v[i] /= sum;
		}
		break;
	}
	return v;
}

MLPClassifier_sklearn::MLPClassifier_sklearn(std::string fpath_model)
{
	FILE *fid = fopen(fpath_model.c_str(), "rb");
	W_hidden = read_matrix(fid);
	B_hidden = read_matrix(fid);
	W_output = read_matrix(fid);
	B_output = read_matrix(fid);
	std::vector<double> actFuncs(2);
	fread(actFuncs.data(), sizeof(double), 2, fid);
	actFuncHidden = map_actFunc_num_to_enum[(int)actFuncs[0]];
	actFuncOutput = map_actFunc_num_to_enum[(int)actFuncs[1]];
	fclose(fid);
}

RowVecMLP MLPClassifier_sklearn::project_onto_hidden_layer(RowVecMLP input_feature_vector)
{
	return apply_actFunc(actFuncHidden, input_feature_vector * W_hidden + B_hidden);
}

RowVecMLP MLPClassifier_sklearn::project_onto_output_layer(RowVecMLP input_feature_vector)
{
	return apply_actFunc(actFuncOutput, project_onto_hidden_layer(input_feature_vector) * W_output + B_output);
}

int MLPClassifier_sklearn::predict_class_label(RowVecMLP input_feature_vector)
{
	RowVecMLP probs_pred = project_onto_output_layer(input_feature_vector);

	int num_output_neurons = probs_pred.size();

	if (num_output_neurons == 1)
	{
		if (probs_pred[0] > 0.5)
			return 1;
		return 0;
	}
	else
	{
		int classIdx = 0;
		for (int i = 0; i < num_output_neurons; i++) {
			classIdx = probs_pred[i] > probs_pred[classIdx] ? i : classIdx;
		}
		return classIdx;
	}
}

RowVecMLP MLPClassifier_sklearn::predict_class_prob(RowVecMLP input_feature_vector)
{
	return project_onto_output_layer(input_feature_vector);
}
