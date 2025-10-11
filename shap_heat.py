import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

g218_125 = pd.read_csv('1.25.csv')
g218_250 = pd.read_pickle('218_250')
shap_values_125 = np.load('shap_values_125.npy')
shap_values_250 = np.load('shap_values_250.npy')

shap_values_125 = pd.DataFrame(shap_values_125)
shap_values_250 = pd.DataFrame(shap_values_250)
shap_values_125.index = g218_250.index
shap_values_125.columns = g218_125.columns
shap_values_250.index = g218_250.index
shap_values_250.columns = g218_250.columns
g218_125.index = g218_250.index

common_columns = shap_values_125.columns.intersection(shap_values_250.columns)
shap_values_250 = shap_values_250[common_columns]

corr_125 = shap_values_125.corr()
corr_250_commen = shap_values_250.corr()

mean_corr_125 = np.mean(corr_125, axis=0)
mean_corr_250_commen = np.mean(corr_250_commen, axis=0)
mean_shap_values_125 = np.mean(shap_values_125, axis=0)
mean_shap_values_250 = np.mean(shap_values_250, axis=0)

df = pd.DataFrame({
    'mean_shap_values_125': mean_shap_values_125,
    'mean_shap_values_250': mean_shap_values_250})
df.to_csv('mean_shap.csv')

plt.figure(figsize=(8, 6))
sns.boxplot(data=df)

plt.title('Comparison of SHAP values distribution between 1.25% and 2.50% datasets')
plt.ylabel('SHAP value')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df)
plt.title('Comparison of SHAP values distribution between 1.25% and 2.50% datasets')
plt.ylabel('SHAP value')
current_ticks = plt.gca().get_yticks()
new_ticks = [tick if tick != 0.02 else 0.01 for tick in current_ticks]
new_ticks = [tick if tick != -0.02 else 0.02 for tick in new_ticks]
plt.yticks(new_ticks)
plt.show()

