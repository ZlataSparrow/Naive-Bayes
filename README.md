### 🎬 Naive Bayes & Movie Recommendation System

This repository explores the Naive Bayes classification algorithm: from mathematical foundations to a real-world application in building a movie recommender system.

⸻

#### Results: 

1. Best model achieved an AUC of ~0.72, indicating good discrimination between “like” and “dislike”.
2. Demonstrated how smoothing (alpha) and prior learning (fit_prior) impact performance.
3. Visual analysis confirmed that moderate smoothing with true priors gave the most stable results.


#### Project Overview

The project is split into two notebooks:

1. `01_Naive_Bayes.ipynb`

* Bayes’ Theorem and Naive Bayes classifiers - theory
* How to calc prior, likelihood, and posterior probabilities
* A simple binary classifier using add-one (Laplace) smoothing.
* How changing the smoothing factor (alpha) affects predictions.

3. `02_A_Movie_Recom_Naive_Bayes.ipynb`
   
* Multinomial Naive Bayes to predict whether a user will “like” or “dislike” a movie (rating > 3).
* User–movie rating matrix as input features.
* Hyperparameter tuning for alpha (smoothing) and fit_prior (class priors) using Stratified K-Fold Cross-Validation.
* Performance evaluation via: ROC Curve & AUC Score, Precision, Recall, F1-score, and Accuracy
* Visualization of results to find the best parameter combination for optimal model performance

⸻

 #### Key Concepts
 
 * Bayes’ Theorem
 * Laplace (Add-One) Smoothing to prevent zero probabilities.
 * Model evaluation metrics: ROC–AUC, F1-score, Precision, Recall.
 * Hyperparameter tuning via cross-validation.

⸻

#### Tools & Libraries
	•	Python, NumPy, Pandas
	•	Scikit-learn (MultinomialNB, StratifiedKFold, roc_auc_score, etc.)
	•	Matplotlib, Seaborn


