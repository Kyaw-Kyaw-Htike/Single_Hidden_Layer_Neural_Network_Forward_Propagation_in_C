// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include <vector>
#include <iostream>
#include "MLPClassifier_sklearn.h"

using namespace std;

int main(int argc, const char * argv[]) {
			
	MLPClassifier_sklearn mlpobj("D:/Working_on_currently/one_Hidden_Layer_Neural_Network_Forward_Propagation_in_C/model_keras.bin");
	//MLPClassifier_sklearn mlpobj("D:/Working_on_currently/one_Hidden_Layer_Neural_Network_Forward_Propagation_in_C/model.bin");

	const int ndims = 4;

	/* Features: */
	RowVecMLP input_fvec(ndims);
	input_fvec[0] = 5.9;
	input_fvec[1] = 3.0;
	input_fvec[2] = 5.1;
	input_fvec[3] = 1.8;

	RowVecMLP probs_pred = mlpobj.predict_class_prob(input_fvec);
	int label_class = mlpobj.predict_class_label(input_fvec);
	printf("num output neurons = %d\n", probs_pred.size());
	printf("Predicted class label = %d\n", label_class);
	printf("Predicted class probabilities: ");
	for (size_t i = 0; i < probs_pred.size(); i++)
	{
		printf("%f, ", probs_pred[i]);
	}
	printf("\n");

	return 0;
	
}


