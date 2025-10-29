### 🎬 Naive Bayes & Movie Recommendation System

This repository explores the Naive Bayes classification algorithm: from mathematical foundations to a real-world application in building a movie recommender system.

⸻

#### Results: 

1. Baseline Model (02_A_Movie_Recom_Naive_Bayes)
* Best Multinomial Naive Bayes achieved an AUC of ~0.72, showing good separation between “like” and “dislike” classes.
* Demonstrated how smoothing (alpha) and prior learning (fit_prior) affect model stability.
* Visual analysis confirmed that moderate smoothing (alpha ≈ 1) with learned priors delivered the most reliable results.

2. Hybrid Model (03_A_Movie_Recom_Hybrid_Naive_Bayes)
* The baseline user–movie matrix was extended with user demographics (age, gender, occupation) and movie genres.
* Achieved a stable AUC of ~0.59, showing that while richer features improved interpretability, Naive Bayes reached its performance ceiling due to feature correlations.
* Cross-validation confirmed that changes in alpha and fit_prior had minimal impact, indicating the model fully utilized the available feature information.

⸻


#### Project Overview

The project is split into two notebooks:

1. `01_Naive_Bayes.ipynb`

* Bayes’ Theorem and Naive Bayes classifiers - theory
* How to calculate prior, likelihood, and posterior probabilities
* A simple binary classifier using add-one (Laplace) smoothing.
* How changing the smoothing factor (alpha) affects predictions.

2. `02_A_Movie_Recom_Naive_Bayes.ipynb`
   
* Multinomial Naive Bayes to predict whether a user will “like” or “dislike” a movie (rating > 3).
* User–movie rating matrix as input features.
* Hyperparameter tuning for alpha (smoothing) and fit_prior (class priors) using Stratified K-Fold Cross-Validation.
* Performance evaluation via: ROC Curve & AUC Score, Precision, Recall, F1-score, and Accuracy
* Visualization of results to find the best parameter combination for optimal model performance


3.	`03_A_Movie_Recom_Hybrid_Naive_Bayes.ipynb`
   
* Extended the model to include user attributes (gender, age, occupation) and movie genres alongside ratings
* One-hot and multi-label encoding for categorical and multi-genre features
* Demonstrated how richer, content-based features can complement collaborative signals
* Stratified K-Fold cross-validation for robustness and AUC stability analysis
* Discussion of Naive Bayes limitations in capturing correlated features and next steps (e.g., Logistic Regression, Random Forest)

⸻

 #### Key Concepts
 
 * Bayes’ Theorem
 * Laplace (Add-One) Smoothing to prevent zero probabilities.
 * Model evaluation metrics: ROC–AUC, F1-score, Precision, Recall.
 * Hyperparameter tuning via cross-validation
 * Feature engineering for hybrid recommender systems (demographics + content)

⸻

#### Tools & Libraries
	•	Python, NumPy, Pandas
	•	Scikit-learn (MultinomialNB, StratifiedKFold, roc_auc_score, etc.)
	•	Matplotlib, Seaborn


