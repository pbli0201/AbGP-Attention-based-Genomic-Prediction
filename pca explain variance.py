import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge

g7412 = pd.read_pickle('snp_1496_mode')
g592929 = pd.read_pickle('snp_1496_592929_mode')

X_full = g592929 
X_subset = g7412 

K = 15 
pca_full = PCA(n_components=K)
pca_full.fit(X_full)
PC_full_scores = pca_full.transform(X_full)

r2_list = []
for i in range(K):
    y = PC_full_scores[:, i] 
    krr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.0001)
    krr.fit(X_subset, y)
    y_pred = krr.predict(X_subset)
    r2 = r2_score(y, y_pred)
    r2_list.append(r2)

plt.figure(figsize=(8, 5))
plt.bar(range(1, K+1), r2_list, color='lightseagreen')
plt.xlabel('Principal Component (PC)')
plt.ylabel('R^2 of Prediction')
plt.title('Explained loci Variance of PCs selected by Attention value')
plt.ylim([0, 1])
plt.xticks(range(1, K+1), labels=[f'K{i}' for i in range(1, K+1)])

for i, v in enumerate(r2_list):
    plt.text(i+1, v + 0.01, f"{v:.2f}", ha='center')
plt.show()

###################################
K = 1496 
pca_full = PCA(n_components=K)
pca_full.fit(X_full)
PC_full_scores = pca_full.transform(X_full) 

r2_list = []
predicted_scores = np.zeros_like(PC_full_scores)

for i in range(K):
    y = PC_full_scores[:, i] 
    
    krr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.0001)
    krr.fit(X_subset, y)
    y_pred = krr.predict(X_subset)
    
    predicted_scores[:, i] = y_pred
    r2 = r2_score(y, y_pred)
    r2_list.append(r2)

plt.figure(figsize=(8, 5))
plt.bar(range(1, K+1), r2_list, color='lightseagreen')
plt.xlabel('Principal Component (PC)')
plt.ylabel('R^2 of Prediction')
plt.title('Explained loci Variance of PCs selected by Attention value')
plt.ylim([0, 1])
plt.xticks(range(1, K+1), labels=[f'PC{i}' for i in range(1, K+1)])
for i, v in enumerate(r2_list):
    plt.text(i+1, v + 0.01, f"{v:.2f}", ha='center')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i in range(6): 
    ax = axes[i]
    ax.scatter(PC_full_scores[:, i], predicted_scores[:, i], alpha=0.6)
    ax.plot(PC_full_scores[:, i], PC_full_scores[:, i], color='red', linestyle='--', label='Ideal Fit (y=x)')
    ax.set_xlabel(f'PC{i+1} Actual')
    ax.set_ylabel(f'PC{i+1} Predicted')
    ax.set_title(f'PC{i+1}: $R^2={r2_list[i]:.2f}$')
    ax.legend()

plt.suptitle('Accuracy of Principal Component Prediction with KRR PC1-PC6')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance of Principal Components')
plt.legend()
plt.show()






