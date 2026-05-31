import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.multiclass import OneVsRestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

data = pd.read_csv("/data/users/PVK/data.csv")
X = data.iloc[:, 0:13].values
Y = data.iloc[:, 14].values  
X_new = np.delete(X, [4, 5], axis=1)
X_n = list(data)[0:13]
X_m = [name for i, name in enumerate(X_n) if i not in [4, 5]]

X_train, X_test, y_train, y_test = train_test_split(
    X_new, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

def optimize_model(pipeline, params, X, y):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring='roc_auc_ovr', 
        cv=StratifiedKFold(5, shuffle=True, random_state=100),
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

svc_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=84)),
    ('scaler', StandardScaler()),
    ('model', SVC(probability=True, decision_function_shape='ovr', random_state=84))
])
svc_params = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf'],
    'model__gamma': ['scale', 0.1, 1]
}
best_svc, svc_best_params = optimize_model(svc_pipe, svc_params, X_train, y_train)
print("\n=== SVC best ===")
print(svc_best_params)

rf_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('model', RandomForestClassifier( random_state=42))
])
rf_params = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [3, 5, None],
    'model__max_features': ['sqrt', 0.8]
}
best_rf, rf_best_params = optimize_model(rf_pipe, rf_params, X_train, y_train)

xgb_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    # 增加 random_state=42
    ('model', XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss', 
        random_state=42))
])
xgb_params = {
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5, 7],          
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0], 
    'model__gamma': [0, 0.1, 1]            
}
best_xgb, xgb_best_params = optimize_model(xgb_pipe, xgb_params, X_train, y_train)

lr_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=55)),
    ('scaler', StandardScaler()),
    ('model', OneVsRestClassifier(LogisticRegression(max_iter=10000, random_state=55))) 
])

lr_params = {
    'model__estimator__C': [0.1, 1, 10],       
    'model__estimator__solver': ['lbfgs', 'saga']
}
best_lr, lr_best_params = optimize_model(lr_pipe, lr_params, X_train, y_train)

custom_weights = {0: 1, 1: 1, 2: 5}
stacking_clf = StackingClassifier(
    estimators=[
        ('svc', best_svc),
        ('rf', best_rf),
        ('xgb', best_xgb), 
        ('lr', best_lr)
    ],
   
    final_estimator=LogisticRegression(class_weight=custom_weights, max_iter=1000, random_state=42),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1,
    passthrough=False
  
)
stacking_clf.fit(X_train, y_train) 
y_score = stacking_clf.predict_proba(X_test)
print(f"X_test shape: {X_test.shape}, y_score shape: {y_score.shape}")

background_data = shap.sample(X_train, 100) 
explainer = shap.KernelExplainer(stacking_clf.predict_proba, background_data)
shap_values = explainer.shap_values(X_new)

target_class = 2
if isinstance(shap_values, list):
    shap_df = pd.DataFrame(shap_values[target_class], columns=X_m)
else:
    shap_df = pd.DataFrame(shap_values, columns=X_m)
    
shap_df.to_csv('shap_values.csv', index=False)

mol_data = pd.read_csv("/data/users/PVK/test.csv")  
X_mol_raw = mol_data.iloc[:, 0:13].values  
X_mol = np.delete(X_mol_raw, [4, 5], axis=1) 

mol_proba = stacking_clf.predict_proba(X_mol)
mol_preds = stacking_clf.predict(X_mol)
mol_proba_class2 = mol_proba[:, target_class]

try:
    mol_names = mol_data['Name'].values
except KeyError:
    mol_names = mol_data.iloc[:, -1].values

mol_results = pd.DataFrame({
    "SampleID": mol_names,  
    "Predicted_Label": mol_preds,
    "Class2_Probability": mol_proba_class2
})
mol_results_sorted = mol_results.sort_values(by="Class2_Probability", ascending=False)
mol_results_sorted.to_csv("1mol_class2_probabilities_named.csv", index=False)

explainer_proba = shap.KernelExplainer(stacking_clf.predict_proba, background_data)
shap_values_mol = explainer_proba.shap_values(X_mol)

for i in range(len(X_mol)):
    pred_class = mol_preds[i]
    class2_prob = mol_proba_class2[i]

    if isinstance(shap_values_mol, list):
        current_shap_value = shap_values_mol[pred_class][i]
        current_base_value = explainer_proba.expected_value[pred_class]
    else:
        current_shap_value = shap_values_mol[i]
        current_base_value = explainer_proba.expected_value
    
    force_plot_html = shap.force_plot(
        base_value=current_base_value,
        shap_values=current_shap_value,
        features=X_mol[i],
        feature_names=X_m,
        matplotlib=False,
        show=False
    )
    shap.save_html(f"MOL_force_plot_sample_{i}_class2_{class2_prob:.2f}.html", force_plot_html)

    plt.figure()
    shap.force_plot(
        base_value=current_base_value,
        shap_values=current_shap_value,
        features=X_mol[i],
        feature_names=X_m,
        matplotlib=True,
        show=False 
    )
    plt.title(f"Sample {i} | Pred: Class {pred_class} | Class 2 Prob: {class2_prob:.2f}", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(f"MOL_force_plot_sample_{i}_class2_{class2_prob:.2f}.png", dpi=300, bbox_inches='tight')
    plt.close()

