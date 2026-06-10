# renewind-turbine-failure-prediction-neuralnetwork-modelling
Classification model that predicts wind turbine generator failures from sensor data for ReneWind, enabling predictive maintenance that minimizes repair, replacement, and inspection costs.
ReneWind — Wind Turbine Generator Failure Prediction

Wind energy is one of the most mature renewable technologies in the global energy mix, and keeping turbines operational is critical to its economics. ReneWind uses machine learning to improve the machinery and processes behind wind energy generation, and has collected sensor data on generator failures across its fleet. The goal is to move from reactive repairs to predictive maintenance — catching failures before they happen, so generators can be repaired (cheap) instead of replaced (expensive).

This project builds, tunes, and compares multiple classification models to predict generator failures from a ciphered, sensor-derived dataset, and selects the model that best minimizes total maintenance cost.

Key objectives

Predict whether a wind turbine generator will fail (1) or not (0) based on sensor readings
Compare multiple classification algorithms and select the best performer
Minimize overall maintenance cost by balancing repair, replacement, and inspection trade-offs
Cost-aware evaluation The choice of metric is driven by business cost, not raw accuracy:

True Positive (TP) — correctly predicted failure → repair cost (low)
False Negative (FN) — missed failure → replacement cost (highest)
False Positive (FP) — false alarm → inspection cost (lowest)
Since missed failures are the costliest outcome, Recall on the failure class is prioritized, with F1-score used to keep false alarms in check.

Dataset

Train.csv — 20,000 observations for model training and tuning
Test.csv — 5,000 observations reserved for final model evaluation
40 ciphered predictor variables + 1 binary target (1 = failure, 0 = no failure)
Approach

Exploratory Data Analysis (EDA) on the transformed sensor features
Missing value treatment and feature scaling
Handling class imbalance using SMOTE and Random UnderSampling
Model building with Logistic Regression, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, and XGBoost
Hyperparameter tuning via GridSearchCV / RandomizedSearchCV on the most promising models
Pipeline construction combining preprocessing and the final estimator for production readiness
Performance comparison on the held-out test set using Recall, F1-score, and Confusion Matrix

Deliverables: Cleaned and resampled datasets, EDA notebook, trained and tuned classification models, production-ready ML pipeline, cost-aware performance comparison, and recommendations for deploying predictive maintenance at ReneWind.

Tech Stack
Programming Language

Python 3
Deep Learning & Modeling

TensorFlow 2.19.0
Keras (Sequential API, Dense, Dropout, BatchNormalization, Input layers)
Optimizers: SGD (with Nesterov momentum), Adam
Regularization: L2 (keras.regularizers), Dropout
Callbacks: ReduceLROnPlateau
Machine Learning & Preprocessing

Scikit-learn 1.6.1
StandardScaler for feature scaling
train_test_split for data partitioning
compute_class_weight for handling class imbalance
classification_report for model evaluation
Data Manipulation & Analysis

Pandas — data loading, cleaning, imputation, and tabular analysis
NumPy — numerical computations and array operations
SciPy (scipy.stats.pointbiserialr) — point-biserial correlation with target
Data Visualization

Matplotlib — loss curves, recall curves, distribution plots
Seaborn — box plots, histograms with KDE, count plots, correlation heatmaps
Development Environment

Jupyter Notebook
Google Colab (with Google Drive integration for dataset access)
Techniques Applied

Exploratory Data Analysis (EDA) — univariate & bivariate analysis
Missing value imputation (median for skewed, mean for symmetric distributions)
Outlier detection using IQR method
Standard scaling of features
Class imbalance handling via class weighting
Neural network architecture experimentation (depth, width, regularization)
Dropout, L2 regularization, and Batch Normalization for generalization
Learning rate scheduling (ReduceLROnPlateau)
Model evaluation using Precision, Recall, F1-score, and classification reports

Quick Mentions:
Python · TensorFlow/Keras · Scikit-learn · Pandas · NumPy · SciPy · Matplotlib · Seaborn · Jupyter Notebook · Google Colab

