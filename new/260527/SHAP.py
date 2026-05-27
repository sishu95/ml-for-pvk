import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.over_sampling import SMOTE 
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# 1. Read Data
data = pd.read_csv("/data/users/lsy/PVK/new/new.csv")
X = data.iloc[:, 0:13].values
Y = data.iloc[:, 14].values 
X_new = np.delete(X, [4, 5], axis=1)
X_n = list(data)[0:13]
X_m = [name for i, name in enumerate(X_n) if i not in [4, 5]]

# 2. Splitting test and training datasets
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
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

# 3. Optimize SVC model
svc_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', SVC(probability=True, decision_function_shape='ovr', class_weight='balanced'))
])
svc_params = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf'],
    'model__gamma': ['scale', 0.1, 1]
}
best_svc, svc_best_params = optimize_model(svc_pipe, svc_params, X_train, y_train)
print("\n=== SVC best ===")
print(svc_best_params)

# 4. Optimize RF model
rf_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(class_weight='balanced'))
])
rf_params = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [3, 5, None],
    'model__max_features': ['sqrt', 0.8]
}
best_rf, rf_best_params = optimize_model(rf_pipe, rf_params, X_train, y_train)

# 5. Optimize XGB model
xgb_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('model', XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss'))
])
xgb_params = {
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5],
    'model__subsample': [0.8, 1.0]
}
best_xgb, xgb_best_params = optimize_model(xgb_pipe, xgb_params, X_train, y_train)

# 6. Optimize LR model
lr_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', OneVsRestClassifier(LogisticRegression(max_iter=10000))) 
])

lr_params = {
    'model__estimator__C': [0.1, 1, 10],       
}
best_lr, lr_best_params = optimize_model(lr_pipe, lr_params, X_train, y_train)

stacking_clf = StackingClassifier(
    estimators=[
        ('svc', best_svc),
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lr', best_lr)
    ],
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=5,
    n_jobs=-1
)

print("正在训练 Stacking 模型...")

stacking_clf.fit(X_train, y_train) 

y_score = stacking_clf.predict_proba(X_test)

y_train_bin = label_binarize(y_train_res, classes=np.unique(y_train_res))
y_test_bin = label_binarize(y_test, classes=np.unique(y_train_res))

print("X_test shape:", X_test.shape)
print("y_score shape:", y_score.shape)

print("正在计算全局 SHAP 值...")

background = shap.sample(X_train, 100) 
explainer = shap.KernelExplainer(stacking_clf.predict, background)
shap_values_all = explainer.shap_values(X_new)

target_class_idx = 2 
if isinstance(shap_values_all, list):
    shap_values_target = shap_values_all[target_class_idx]
else:
    shap_values_target = shap_values_all

columns = [f'feature{i+1}' for i in range(X_new.shape[1])]
X_df = pd.DataFrame(X_new, columns=columns)

shap_df = pd.DataFrame(shap_values_target, columns=X_m)
shap_df.to_csv('shap_values_class2.csv', index=False)
print("全局 SHAP values (Class 2) 已保存到 'shap_values_class2.csv'")

shap.initjs() 
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = "Arial" 
plt.rcParams.update({'font.size': 14}) 

shap.summary_plot(shap_values_target, X_new, plot_type="bar", 
                feature_names=X_m, show=False)
plt.tight_layout()
plt.savefig('shap1_bar.png', dpi=300)
plt.close()  

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_target, X_new, plot_type="dot",
                feature_names=X_m, show=False)
plt.tight_layout()
plt.savefig('shap2_dot.png', dpi=300)
plt.close()  

X = pd.DataFrame(X_new, columns=X_m)

for feature in X_m:
    plt.figure(figsize=(8, 6))  
    plt.rcParams['font.family'] = 'Arial'  
    plt.rcParams['font.size'] = 12 
    
    shap.dependence_plot(
        feature,         
        shap_values_target,    
        X,       
        interaction_index=None,  
        show=False      
    )
    
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('Arial') 
        
    plt.gca().spines['top'].set_visible(True)  
    plt.gca().spines['right'].set_visible(True)  
    plt.gca().spines['bottom'].set_visible(True)  
    plt.gca().spines['left'].set_visible(True)  
    plt.tight_layout()
    plt.savefig(f'dependence_{feature}.png', dpi=300)  
    plt.close()  

print("All dependence plots have been saved.")