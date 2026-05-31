import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import StackingClassifier

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


stacking_clf = StackingClassifier(
    estimators=[
        ('svc', best_svc),
        ('rf', best_rf),
        ('xgb', best_xgb), 
        ('lr', best_lr)
    ],

    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),  
    cv=StratifiedKFold(5, shuffle=True, random_state=42), 
    n_jobs=-1,
    passthrough=False 
)



stacking_clf.fit(X_train, y_train)
y_score = stacking_clf.predict_proba(X_test)

y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

n_classes = y_test_bin.shape[1]
print("X_test shape:", X_test.shape)
print("y_score shape:", y_score.shape)
print("y_test_bin shape:", y_test_bin.shape)

plt.figure(figsize=(10,9))  
target_class = 2 
class_names = np.unique(y_train_res)
class_idx = np.where(class_names == target_class)[0][0]
model_list = [
    ('SVC', best_svc, 'dodgerblue', '--'),
    ('RF', best_rf, 'limegreen', '-.'),
    ('XGB', best_xgb, 'crimson', ':'),
    ('LR', best_lr, 'darkcyan', '--'),
    ('Ensemble', stacking_clf, 'purple', '-')
]
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', 
        label='Random (AUC=0.50)', alpha=0.8)

for name, model, color, ls in model_list:
    try:
        y_score = model.predict_proba(X_test)
        
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, 
                color=color,
                linestyle='-',
                linewidth=2.5,
                alpha=0.9,
                label=f'{name} (AUC={roc_auc:.2f})')
        
    except Exception as e:
        print(f"{name} false: {str(e)}")
        continue

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=20)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)  
plt.xlabel('False Positive Rate', fontsize=32, labelpad=10) 
plt.ylabel('True Positive Rate', fontsize=32, labelpad=10)  
plt.title(f'ROC Curve Comparison (Class {target_class})', fontsize=20, pad=15, weight='bold')  
legend = plt.legend(
    loc='lower right',
    fontsize=22,  
    title='Models',
    title_fontsize=22,
    bbox_to_anchor=(0.98,0.01),  
    borderpad=0.4,                
    handlelength=2.5,             
    handletextpad=0.5,            
    labelspacing=0.8,             
    frameon=False                
)

ax.tick_params(axis='both', which='major', 
              labelsize=25) 

plt.savefig('roc_comparison.png', 
           dpi=300, 
           bbox_inches='tight',
           facecolor='white')
plt.show()
