import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

phe_1715 = pd.read_csv('1715_phe.csv',index_col=0)
phe_adjuest = pd.read_csv("phe_adjuest111.csv")
phe_adjuest.set_index('770K',inplace=True)

for col in phe_adjuest.columns:
    phe_adjuest[col] = phe_adjuest[col].astype('category')
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(phe_adjuest)
phe_adjuest_onehot = pd.DataFrame(encoded_array, index=phe_adjuest.index, 
                                  columns=encoder.get_feature_names_out(phe_adjuest.columns))

common_index = phe_1715.index.intersection(phe_adjuest_onehot.index)
yuanshibiaoxing = phe_1715.loc[common_index]
gudingxiaoying = phe_adjuest_onehot.loc[common_index]

yuanshibiaoxing.replace(' ', np.nan, inplace=True) 
yuanshibiaoxing = yuanshibiaoxing.apply(pd.to_numeric, errors='coerce') 

yuanshibiaoxing1 = pd.DataFrame(yuanshibiaoxing.iloc[:,19])
missing_phe_index = yuanshibiaoxing1[yuanshibiaoxing1.isna().any(axis=1)].index
yuanshibiaoxing2 = yuanshibiaoxing1.drop(index=missing_phe_index)
gudingxiaoying1 = gudingxiaoying.drop(index=missing_phe_index)
    
X = gudingxiaoying1
y = yuanshibiaoxing2
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
        
corrected_values = model.resid
corrected_data = pd.DataFrame(corrected_values, columns=['Corrected_Phenotype'])
corrected_data.to_csv('adjuest_phe20.csv')





