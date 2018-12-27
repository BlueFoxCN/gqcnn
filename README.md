Both train accuracy and validation accuracy reached highest level, where the former is 0.9445 and the latter is 0.9372, at the condition: 1.weights regularization 2.noise 3.preprocess 4.split training set and test set image-wisely 5.the most important: change the base lr from 0.01 to 0.001

Notes:
1.Without adding noise would result in overfitting
2.With base lr as 0.01, adding regularization and preprocess would result cost to NAN
