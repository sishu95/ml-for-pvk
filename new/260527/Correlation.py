import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read Data
data = pd.read_csv("new/ML code/data.csv")
data.head()
X = data.iloc[:, 0:13].values  
Y = data.iloc[:, 14:15].values   
X_n = list(data)[0:13]
Y_n = list(data)[14:15]
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print(X_n)
print(Y_n)
X_df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])

# 2. Using Pearson Correlation
cor = X_df.corr()
mask = np.zeros_like(cor)
plt.rcParams['font.family'] = "Arial"
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12,8))
    ax = sns.heatmap(
        cor, 
        annot=True,
        annot_kws={"size": 14,"fontname": "Arial"},  
        fmt=".2f",
        cmap=plt.cm.RdBu_r,
        linewidths=0,
        vmin=-1,
        vmax=1,
        mask=mask, 
        square=True,
        xticklabels=X_n, 
        yticklabels=X_n,
        cbar_kws={"shrink": 0.8, "pad": 0.02}  
    )
    plt.setp(ax.get_xticklabels(), 
             rotation=45,  
             ha="right",  
             fontsize=22, 
             fontname="Arial")  
    
    plt.setp(ax.get_yticklabels(), 
             fontsize=22, 
             fontname="Arial")  

# 3. Print and Save Figure
plt.title("Correlation Matrix", fontsize=32,fontname="Arial")
plt.tight_layout()  
plt.savefig('Correlation Matrix.png')
plt.show()