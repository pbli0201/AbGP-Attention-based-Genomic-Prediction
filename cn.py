import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


g7412 = pd.read_pickle('snp_pig')

props = np.arange(0.05, 1.01, 0.05)
cond_nums_g7412 = []

for p in props:
    n_sample = int(g7412.shape[0] * p)
    sub_g7412 = g7412.iloc[:n_sample, :]
    # sub_g592929 = g592929.iloc[:n_sample, :]
    c_g7412 = np.linalg.cond(sub_g7412.to_numpy())
    # c_g592929 = np.linalg.cond(sub_g592929.to_numpy())
    
    cond_nums_g7412.append(c_g7412)
    # cond_nums_g592929.append(c_g592929)

plt.figure(figsize=(8, 6))
plt.plot(props * 100, cond_nums_g7412, marker='o', color='blue', label='1.25')
plt.xlabel('Sample Size (%)')
plt.ylabel('Condition Number')
plt.legend()
plt.show()

cond_nums_g7412 = pd.DataFrame(cond_nums_g7412)
cond_nums_g7412.to_csv('cn14823.csv', index=False)

