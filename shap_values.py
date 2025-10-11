import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

g218_125 = pd.read_csv('1.25.csv')
g218_250 = pd.read_pickle('218_250')
phe = pd.read_pickle('adjusted_phe')
phe = phe.iloc[:, 1]

model_1_25 = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model_1_25.fit(g218_125, phe)

explainer_1_25 = shap.TreeExplainer(model_1_25)
shap_values_1_25 = explainer_1_25.shap_values(g218_125)

model_2_50 = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model_2_50.fit(g218_250, phe)

explainer_2_50 = shap.TreeExplainer(model_2_50)
shap_values_2_50 = explainer_2_50.shap_values(g218_250)

selected_features = []
for i in range(shap_values_1_25.shape[1]):
    if np.mean(shap_values_1_25[:, i]) > 0:
        shap_1_25_positive = np.sum(shap_values_1_25[:, i] > 0)
        shap_2_50_positive = np.sum(shap_values_2_50[:, i] > 0)
        if shap_2_50_positive < shap_1_25_positive:
            selected_features.append(i)

shap.summary_plot(shap_values_1_25[:, selected_features], g218_125.iloc[:, selected_features], title='SHAP Summary Plot for 1.25% Data Set')
shap.summary_plot(shap_values_2_50[:, selected_features], g218_250.iloc[:, selected_features], title='SHAP Summary Plot for 2.50% Data Set')

def plot_dependence(feature):
    plt.figure()
    shap.dependence_plot(feature, shap_values_1_25, g218_125, feature_names=g218_125.columns, show=False)
    plt.title(f'Dependence Plot for Feature {g218_125.columns[feature]} in 1.25% Data Set')
    plt.savefig(f'dependence_1_25_feature_{feature}.png')
    plt.close()

    plt.figure()
    shap.dependence_plot(feature, shap_values_2_50, g218_250, feature_names=g218_250.columns, show=False)
    plt.title(f'Dependence Plot for Feature {g218_250.columns[feature]} in 2.50% Data Set')
    plt.savefig(f'dependence_2_50_feature_{feature}.png')
    plt.close()

with ThreadPoolExecutor() as executor:
    executor.map(lambda feature: plot_dependence(feature, shap_values_1_25, g218_125, shap_values_2_50, g218_250), selected_features)

    
shap.dependence_plot('variant237115', shap_values_1_25, g218_125, feature_names=g218_125.columns)

shap.dependence_plot('variant339600', shap_values_1_25, g218_125, 
                     interaction_index='variant339600', feature_names=g218_125.columns,show=False)
plt.xlabel('')
plt.show()

shap.dependence_plot('variant339600', shap_values_1_25, g218_125, 
                     interaction_index='variant339600', feature_names=g218_125.columns,show=False)
plt.ylim(-0.6, 0.4) 
plt.yticks(np.arange(-0.6, 0.6, 0.2))  

plt.xlabel('')
plt.show()

shap.dependence_plot('variant339600', shap_values_2_50, g218_250, 
                     interaction_index='variant339600', feature_names=g218_250.columns,show=False)
plt.ylim(-0.6, 0.4) 
plt.yticks(np.arange(-0.8, 0.8, 0.2))  

plt.xlabel('')
plt.show()

