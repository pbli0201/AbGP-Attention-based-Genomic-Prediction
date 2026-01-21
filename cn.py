import numpy as np
import pandas as pd
from numpy.linalg import cond
import matplotlib.pyplot as plt

snpdata = pd.read_pickle('snp_top_2.5_percent.pkl')
snpdata = snpdata.values
n_samples, n_snps = snpdata.shape
cond_numbers = []

for i in range(1, 21):
    sample_size = int(i * n_samples / 20)
    subset = snpdata[:sample_size, :] 
    cov_matrix = np.cov(subset, rowvar=False) 
    condition_number = cond(cov_matrix)
    cond_numbers.append([sample_size, condition_number])

cond_df = pd.DataFrame(cond_numbers, columns=['Sample Size', 'Condition Number'])
cond_df.to_csv('2.5_condition_numbers.csv', index=False)

sample_percentages = np.arange(5, 101, 20)
plt.plot(sample_percentages, cond_df['Condition Number'], marker='o')
plt.grid(True)
plt.show()
