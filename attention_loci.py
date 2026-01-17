import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import pytorch_lightning as pl
from scipy.special import softmax

pl.seed_everything(9527)

Genotype = pd.read_pickle('cattle_218_genotype')
g_cosine = pd.read_pickle('g_cosine')
t_cosine = pd.read_pickle('t_matrix')
g_cosine.index = t_cosine.index
g_matrix = pd.read_pickle('g_matrix')

Q = np.dot(g_matrix, Genotype)
K = np.dot(g_cosine, Genotype)
K_T = K.T
V = np.dot(t_cosine, Genotype)

Attention_scores = np.dot(Q, K_T) / np.sqrt(592929)
Attention_weigts_raw = np.dot(Attention_scores, V)

AW_colmean_raw = np.mean(Attention_weigts_raw, axis=0)
AW_final = softmax(AW_colmean_raw)

AW = pd.DataFrame(AW_final, columns=['Values']).T

Attention_scores = pd.DataFrame(Attention_scores)
Attention_weigts = pd.DataFrame(Attention_weigts_raw)

Attention_scores.to_pickle('Attention_scores.pkl')
Attention_weigts.to_pickle('Attention_weigts.pkl')
AW.to_pickle('Mean_Attention_weigts.pkl')

values = AW.values.flatten()

thresholds = np.arange(100, 96, -0.05)
mean_correlations = []

for threshold in thresholds:
    current_threshold = np.percentile(values, threshold)
    indices = np.where(values >= current_threshold)[0]

    filtered_genotype = Genotype.iloc[:, indices]
    Phenotype = pd.read_csv('iid_218.csv')
    Phe1 = Phenotype.iloc[:, 1]

    correlations = []

    for i in range(2000):
        X_train, X_test, y_train, y_test = train_test_split(
            filtered_genotype, Phe1, test_size=0.2)

        krr = KernelRidge(kernel='rbf')
        krr.fit(X_train, y_train)
        y_pred = krr.predict(X_test)

        if len(np.unique(y_test)) == 1 or len(np.unique(y_pred)) == 1:
            correlation = np.nan
        else:
            correlation, _ = pearsonr(y_test, y_pred)
        correlations.append(correlation)

    mean_correlation = np.nanmean(correlations)
    mean_correlations.append(mean_correlation)
    print(f'Threshold {threshold:.1f}: Mean Pearson Correlation Coefficient: {mean_correlation:.3f}')

print("All Mean Pearson Correlation Coefficients for each threshold:", mean_correlations)
