import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.linear_model import  Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV  

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("/data/users/lsy/PVK/new/260527/data.csv")
X = data.iloc[:, 0:13].values
Y = data.iloc[:, 14].values  
X_new = np.delete(X, [4, 5], axis=1)
X_n = list(data)[0:13]
X_m = [name for i, name in enumerate(X_n) if i not in [4, 5]]


X_train, X_test, y_train, y_test = train_test_split(
    X_new, Y,
    test_size=0.2,
    random_state=42
)

def optimize_regression_model(pipeline, params, X, y, model_name=""):

    n_jobs = -1

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring='neg_mean_squared_error',  
        cv=KFold(5, shuffle=True, random_state=42),
        n_jobs=n_jobs,  
        verbose=1,
        error_score='raise'
    )
    grid.fit(X, y)
    
    return grid.best_estimator_, grid.best_params_

svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVR(max_iter=10000))  
])
svr_params = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf'],
    'model__gamma': ['scale', 0.1],
    'model__epsilon': [0.1, 0.2]
}
best_svr, svr_best_params = optimize_regression_model(svr_pipe, svr_params, X_train, y_train, "SVR")

rf_pipe = Pipeline([
    ('model', RandomForestRegressor(random_state=42, n_jobs=1))  
])
rf_params = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}
best_rf, rf_best_params = optimize_regression_model(rf_pipe, rf_params, X_train, y_train, "Random Forest")

gbr_pipe = Pipeline([
    ('model', GradientBoostingRegressor(random_state=42))
])
gbr_params = {
    'model__learning_rate': [0.05, 0.1],
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__subsample': [0.8, 1.0]
}
best_gbr, gbr_best_params = optimize_regression_model(gbr_pipe, gbr_params, X_train, y_train, "Gradient Boosting")

ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(max_iter=10000))
])
ridge_params = {
    'model__alpha': [0.001, 0.01, 0.1, 1, 10],
}
best_ridge, ridge_best_params = optimize_regression_model(ridge_pipe, ridge_params, X_train, y_train, "Ridge Regression")

ensemble_regressor = StackingRegressor(
    estimators=[
        ('svr', best_svr),
        ('rf', best_rf),
        ('gbr', best_gbr),
        ('ridge', best_ridge)
    ],
    final_estimator=RidgeCV(), 
    cv=5, 
    n_jobs=-1,
    passthrough=False 
)

ensemble_regressor.fit(X_train, y_train)

models = {
    'SVR': best_svr,
    'Random Forest': best_rf,
    'Gradient Boosting': best_gbr,
    'Ridge': best_ridge,
    'Ensemble': ensemble_regressor
}

all_predictions = {}
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    all_predictions[name] = y_pred
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    
    if (y_test != 0).all(): 
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) 
    else:
        mape = np.nan
    
    results[name] = {
        'RMSE': rmse,
        'R² Score': r2,
    }
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 

metrics_to_plot = [ 'RMSE', 'R² Score']
model_names = list(models.keys())
colors = ['dodgerblue', 'limegreen', 'crimson', 'darkcyan', 'purple']

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    values = []
    valid_models = []
    valid_colors = []
    
    for j, model in enumerate(model_names):
        if metric in results[model] and isinstance(results[model][metric], (int, float)):
            values.append(results[model][metric])
            valid_models.append(model)
            valid_colors.append(colors[j])
    
    if values: 
        bars = ax.bar(valid_models, values, color=valid_colors, alpha=0.8)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 0.01 * max(values) if height >= 0 else -0.01 * min(values)
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{value:.4f}', ha='center', va=va, fontsize=10, fontweight='bold')
        
        ax.set_title(f'{metric} Comparison', fontsize=14, weight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
     
        if metric in ['R² Score']:
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

text_str = f'R² = {results["Ensemble"]["R² Score"]:.4f}\nRMSE = {results["Ensemble"]["RMSE"]:.4f}'
ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Regression Model Performance Comparison', fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('regression_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

best_params_summary = {
    'SVR': svr_best_params,
    'Random Forest': rf_best_params,
    'Gradient Boosting': gbr_best_params,
    'Ridge Regression': ridge_best_params
}

for model_name, params in best_params_summary.items():
    print(f"\n{model_name}:")
    for param, value in params.items():
        print(f"  {param}: {value}")

import json
with open('best_model_parameters.json', 'w') as f:
    json.dump(best_params_summary, f, indent=2)
