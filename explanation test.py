import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


phe_1496 = pd.read_csv('adjuested_phe_1496.csv', index_col=0)
phe_1496.replace(' ', np.nan, inplace=True)
phe_1496 = phe_1496.apply(pd.to_numeric, errors='coerce')
Genotypes_1496 = pd.read_pickle('snp_1496_mode')

Phenotypes = pd.DataFrame(phe_1496.iloc[:,3])
missing_phe_index = Phenotypes[Phenotypes.isna().any(axis=1)].index
Genotypes = Genotypes_1496.drop(index=missing_phe_index)
phe = Phenotypes.drop(index=missing_phe_index)
rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

thresholds = [i * 0.05 for i in range(1, 21)]
subset_sizes = [int(len(Genotypes) * t) for t in thresholds]

train_errors = []
test_errors = []
feature_importance_variances = []
conf_interval_stds = []
biases = []
variances = []

for size in subset_sizes:
    subset_genotypes = Genotypes.sample(n=size, random_state=42)
    subset_phe = phe.loc[subset_genotypes.index]

    X_train, X_test, y_train, y_test = train_test_split(
        subset_genotypes, 
        subset_phe, 
        test_size=0.2, 
        random_state=42
    )
    y_train = y_train.values.flatten()
    y_test  = y_test.values.flatten()


    n_runs = 10
    predictions_test_runs = []
    train_errors_runs = []
    test_errors_runs  = []

    for run in range(n_runs):
        rf_model.set_params(random_state=42 + run)
        rf_model.fit(X_train, y_train)

        train_pred = rf_model.predict(X_train)
        train_errors_runs.append(mean_squared_error(y_train, train_pred))

        test_pred = rf_model.predict(X_test)
        test_errors_runs.append(mean_squared_error(y_test, test_pred))
        predictions_test_runs.append(test_pred)

    predictions_test_runs = np.array(predictions_test_runs)

    train_error = np.mean(train_errors_runs)
    test_error  = np.mean(test_errors_runs)
    
    avg_preds_test = np.mean(predictions_test_runs, axis=0) 
    bias = np.mean((avg_preds_test - y_test) ** 2)

    variance = np.mean(np.var(predictions_test_runs, axis=0))

    conf_interval_std = np.mean(np.std(predictions_test_runs, axis=0))


    importances = []
    for run in range(n_runs):
        rf_model.set_params(random_state=100 + run)  
        rf_model.fit(X_train, y_train)
        importances.append(rf_model.feature_importances_)

    importances = np.array(importances) 
    feature_importance_variance = np.mean(np.var(importances, axis=0))

    train_errors.append(train_error)
    test_errors.append(test_error)
    biases.append(bias)
    variances.append(variance)
    conf_interval_stds.append(conf_interval_std)
    feature_importance_variances.append(feature_importance_variance)

df = pd.DataFrame({
    'Sample Size (%)': [t*100 for t in thresholds],
    'Train Error': train_errors,
    'Test Error': test_errors,
    'Bias': biases,
    'Variance': variances,
    'Confidence Interval Std': conf_interval_stds,
    'Feature Importance Variance': feature_importance_variances
})

df.to_csv('random_forest_results——GPT.csv', index=False)
print(df)


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['Sample Size (%)'], df['Train Error'], marker='o', label='Train Error', color='b')
plt.plot(df['Sample Size (%)'], df['Test Error'], marker='o', label='Test Error', color='r')
plt.xlabel("Sample Size (%)")
plt.ylabel("Mean Squared Error")
plt.title("Training vs. Test Error vs. Sample Size")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(df['Sample Size (%)'], df['Feature Importance Variance'], marker='s', label='Feature Importance Variance', color='g')
plt.xlabel("Sample Size (%)")
plt.ylabel("Feature Importance Variance")
plt.title("Feature Importance Variance vs. Sample Size")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(df['Sample Size (%)'], df['Confidence Interval Std'], marker='^', label='Prediction CI Std', color='orange')
plt.xlabel("Sample Size (%)")
plt.ylabel("Confidence Interval Std")
plt.title("Prediction Confidence Interval vs. Sample Size")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(df['Sample Size (%)'], df['Bias'], marker='x', label='Bias', color='blue')
plt.plot(df['Sample Size (%)'], df['Variance'], marker='x', label='Variance', color='red')
plt.xlabel("Sample Size (%)")
plt.ylabel("Error")
plt.title("Bias vs. Variance")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
