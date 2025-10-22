# Collaborative Filtering

This repository presents our project on **Collaborative Filtering**, completed as part of the IASD program (PSL University).  
It explores and compares several methods for predicting user ratings on movies using **Matrix Factorization**, **Kernel Methods**, and **Principal Component Analysis (PCA)**.

---

## Overview

Collaborative filtering is a classical problem in recommender systems, where we aim to predict a user’s rating for unseen items based on historical ratings.  
We implemented and compared multiple algorithms to understand how modeling choices — linear vs. non-linear, bias inclusion, kernelization, and orthogonal constraints — affect predictive performance.

**Main methods explored:**
1. Alternating Least Squares (ALS)  
2. Stochastic Gradient Descent (SGD)  
3. Kernelized Matrix Factorization (Kernel MF)  
4. Principal Component Analysis (PCA and EM-PCA)

Our main evaluation metric was **Root Mean Squared Error (RMSE)** and **accuracy**.

---

## Methods

### 1. Alternating Least Squares (ALS)
- Deterministic and interpretable factorization method.
- Converges quickly due to closed-form updates.

### 2. Stochastic Gradient Descent (SGD)
- Gradient-based incremental optimization for large sparse datasets.
- More flexible but requires careful tuning (learning rate, regularization).
- Achieved smoother convergence and **lower RMSE than ALS** after hyperparameter search.

### 3. Kernelized Matrix Factorization (Kernel MF)
- Extends MF with non-linear kernels to capture complex user–item relations.
- Implemented linear, RBF, and sigmoid kernels.
- Added **genre-based item kernel** using cosine similarity on multi-hot genre vectors.
- Performed **ensemble learning on the simplex**, linearly combining predictions.
- Best ensemble RMSE ≈ **0.896**.

### 4. Principal Component Analysis (PCA)
- Reformulates collaborative filtering as an orthogonal matrix factorization.
- Handles missing ratings by partial covariance estimation.
- Best results at **k = 8 components** with RMSE = **0.92** and Accuracy = **0.24**.
- PCA components were interpreted through correlations with movie genres.

### 5. PCA Improvements
- Implemented **Iterative EM-PCA** for missing data.
- Explored smooth vs. hard rounding schemes.

---

## Key Takeaways

- **ALS** provides a strong deterministic baseline but limited flexibility.  
- **SGD** offers better performance with proper regularization and tuning.  
- Adding **bias terms** to MF significantly improves RMSE.  
- **Kernel methods** bring moderate improvements through non-linear similarity.  
- **PCA** acts as a robust dimensionality reduction approach, yielding interpretable components tied to movie genres.

---

## Repository Structure
├── KMF/ # Kernelized Matrix Factorization implementations
├── PCA/ # PCA, EM-PCA, and KPCA implementations
├── MF_ALS.py # Alternating Least Squares method
├── generate.py # Data generation and preprocessing scripts
├── evaluation.py # Evaluation and metric computation
├── requirements.txt # Python dependencies
├── report.pdf # Full written report
├── slides.pdf # Presentation slides
└── LICENSE


---
## Authors

- Rayane Dakhlaoui — Télécom Paris / IASD PSL

- Nathan Rouillé — Télécom Paris / IASD PSL

- Sacha Khosrowshahi — Télécom Paris / IASD PSL

