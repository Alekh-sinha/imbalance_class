# imbalance_class
This project gives some insight on how to deal with imbalance class.
It can be dealt in two ways:
1.Pre-proessing
2.Adjusting algorithm
Feature-engineering
For an imbalance problem, it is very difficult to prepare a model which is not biased towards negative classs. I tried a few feature engineering option to prepare trustable model. Some of those were:-
•	To use as it is and adjust imbalance by using class weight
•	To use oversampler
•	To use undersampler
•	To use smote    
I fixed my algorithm as decision tree for all these method and tried all of these one by one. I checked for the F1  and ROC_AUC score and found that smote works the best. Smote had the highest F1 score. Input data was divided into test and train data. Test data contained 30% of the data. F1 scored was calculated on  test data and based on that smote was selected.
Algorithm Selection
My strategy for algorithm selection was to try with different algorithms and evaluate on the basis of F1 score and ROC_AUC score. Algorithm for which F1 score was highest was selected. 
Algorithm tried by me was:-
•	Logistic regression with class weights as it was imbalanced class
•	Decision tress with class weights as it was imbalanced class
•	Adaboost as it can deal with imbalanced class
•	Random forest performs well even on imbalanced class
•	XGBoost as it gives a good performance with imbalanced class
Randomsearchcv was used for parameter tuning to get best from all these algorithms. For logistic regression C (inverse of regularization) and penalty (l1 or l2) was adjusted. For decision tree min sample split and depth was adjusted. In adaboost same parameter as of decision tree and along with that n_estimator was adusted. In random forest same parameter as of decision tree and along with number of trees was also adjusted.   For XGBoost learning rate, max depth and n estimator was tuned.
On the basis of these parameter XGBoost was chosen.
Learning:
•	30 % of data was kept for test. This was way too much for imbalanced class
•	More parameters could have been tuned.
•	ADASYN (Adaptive synthetic sampling approach for imbalanced learning) and also other sampling technique could have been explored
