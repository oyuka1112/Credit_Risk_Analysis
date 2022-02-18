# Credit_Risk_Analysis
- Python: sklearn, pandas, and imblearn
- Supervised Learning: Random Forest Classififer, Easy Ensemble AdaBoost Classifier, Random Over Sampler, SMOTE, Clister Centroids, and SMOTEEN
## Overview of the Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, I needed to employ different techniques to train and evaluate models with unbalanced classes. I used imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once I'm done, I'll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk. 
## Results
After preparing the data and transforming it for ready to machine learning. Split the dataset with 25, 75 (default) split.
1. RandomOverSampler (Oversampling)
![](https://user-images.githubusercontent.com/64121596/154624779-2f49d78a-ce80-473e-a8a4-f5763490c730.png)
- balanced accuracy score: 0.66   
- precision: 1
- recall scores: 0.67
2. SMOTE Oversampling)
![](https://user-images.githubusercontent.com/64121596/154624934-cf8e47f6-69e7-4346-a89f-1da0c3ffb81c.png)
- balanced accuracy score: 0.63  
- precision: 1
- recall scores: 0.73
3. ClusterCentroids (Undersampling)
![](https://user-images.githubusercontent.com/64121596/154624969-c3816855-6732-4931-9baa-0fcf33b11a49.png)
- balanced accuracy score: 0.48 
- precision: 0.99
- recall scores: 0.73
4. SMOTEENN (Over-undersampling)
![](https://user-images.githubusercontent.com/64121596/154625035-2057eace-bfb0-4f30-9087-c2f809bca2de.png)
- balanced accuracy score: 0.62  
- precision: 1
- recall scores: 0.62
5. BalancedRandomForestClassifier
![](https://user-images.githubusercontent.com/64121596/154625078-a0ec67c0-b341-41f6-899d-c7d8281c1edd.png)
- balanced accuracy score: 0.76  
- precision: 1
- recall scores: 0.91
- Top 5 important features: 
(0.08969665458209104, 'total_rec_prncp')
 (0.06436823464037578, 'total_rec_int')
 (0.06362653031376969, 'total_pymnt_inv')
 (0.053061501841593216, 'last_pymnt_amnt')
 (0.05184437542411125, 'total_pymnt')
6. EasyEnsembleClassifier
![](https://user-images.githubusercontent.com/64121596/154625109-87b353c2-1ea3-4cc0-af1c-5f8fb2a21dd6.png)
- balanced accuracy score: 0.9 
- precision: 1
- recall scores: 0.96
## Summary
All of the machine learning technique above has precision 1, which means that the false negative is 0. In other words, if a loan was not risky, we predicted everything correctly. In recall, ensemble methods have the highest 0.91 and 0.96. That means we predicted the risky loan with 91% and 96% correctly.  Balanced accuracy score is highest in Easy 
Ensemble Classifier. Therefore, the model ensemble classififer is a good method in this case for detecting a risky loans. 

