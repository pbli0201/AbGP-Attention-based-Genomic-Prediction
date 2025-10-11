import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

g7412 = pd.read_pickle('snp_1496_mode')
g592929 = pd.read_pickle('snp_1496_592929_mode')

pca_full = PCA()
pca_full.fit(g592929)
cum_var_full = np.cumsum(pca_full.explained_variance_ratio_)

pca_sample = PCA()
pca_sample.fit(g7412)
cum_var_sample = np.cumsum(pca_sample.explained_variance_ratio_)

plt.plot(cum_var_full, label='Full Dataset')
plt.plot(cum_var_sample, label='Sampled Dataset (1.25%)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.show()

