import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
import pylab

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from scipy import exp

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Import data

df = pd.read_csv('F:/MSFE/IE517 MLF/group/MLF_GP2_EconCycle.csv', index_col='Date')
df

# Preprocessing
# Finding NA in Dataset
print(df[df.isna().any(axis = 1)])
df=df.dropna(how='any',axis=0)
df.info()
df.describe()
# EDA Part
# Part1 Q-Q Plot

scipy.stats.probplot(df['T1Y Index'], dist = 'norm', plot = pylab)
pylab.title('T1Y Index Probability Plot')
pylab.show()

# EDA Part
# Part2 heat map

cols = ['T1Y Index', 'T2Y Index', 'T3Y Index', 'T5Y Index', 
        'T7Y Index', 'T10Y Index','CP1M','CP3M','CP6M','CP1M_T1Y',
        'CP3M_T1Y','CP6M_T1Y','PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD']

plt.figure(figsize = (12, 12))
cm =np.corrcoef(df[cols].values.T)

sns.set(font_scale = 1)
hm = sns.heatmap(cm, cbar = True, annot = False, square = True, 
                fmt = '.2f', annot_kws = {'size': 20}, yticklabels = cols, 
                xticklabels = cols)

plt.show()



# EDA Part
# Part3 Scatterplot & Histogram
plt.figure(figsize = (15, 15))
sns.pairplot(df[cols], height = 2)
plt.tight_layout()
plt.show()


# EDA Part
# Part4 Boxplot

plt.figure(figsize = (30, 15))

f = df.boxplot(['T1Y Index', 'T2Y Index', 'T3Y Index', 'T5Y Index', 
                'T7Y Index', 'T10Y Index','CP1M','CP3M','CP6M','CP1M_T1Y',
                'CP3M_T1Y','CP6M_T1Y','PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD'])
plt.xlabel("Attribute Index")
plt.show()

# Preprocessing
X = df.iloc[:, 0:12].values
y = df.iloc[:, 13:16].values
y_3m = y[:,0]
y_6m = y[:,1]
y_9m = y[:,2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

y_3m_train=y_train[:,0]
y_3m_test=y_test[:,0]
y_6m_train=y_train[:,1]
y_6m_test=y_test[:,1]
y_9m_train=y_train[:,2]
y_9m_test=y_test[:,2]
# Standization
scaler = StandardScaler()
scalery= StandardScaler()
scaler.fit(df)
df_std = scaler.transform(df)


X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)
y_train_std=scalery.fit_transform(y_train)
y_test_std=scalery.transform(y_test)
y_3m_train_std=y_train_std[:,0]
y_3m_test_std=y_test_std[:,0]
y_6m_train_std=y_train_std[:,1]
y_6m_test_std=y_test_std[:,1]
y_9m_train_std=y_train_std[:,2]
y_9m_test_std=y_test_std[:,2]

#PCA
# PCA Explained Variance plot
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp =[ (i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
var_exp =np.array( [(i / tot) for i in sorted(eigen_vals, reverse=True)])
plt.bar(range(1, 13), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 13), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
#PCA select 3 features
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print("The cumulative explained variance of all features:")
print(pca.explained_variance_ratio_)
print("")

#Linear regression models and evaluation
def train_and_evaluate(clf,X_train_std,X_test_std,y_train_std,y_test_std,linear_or_not):
    clf.fit(X_train_std,y_train_std)
    y_train_pred = clf.predict(X_train_std)
    y_test_pred = clf.predict(X_test_std)
    plt.scatter(y_train_pred,  y_train_pred - y_train_std,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test_std,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.title('Residual errors')
    plt.hlines(y=0, xmin=-1, xmax=1, color='black', lw=2)
    plt.xlim([-1, 1])
    plt.tight_layout()
    plt.show()
    if(linear_or_not):
        print('Slope: ' ,end=' ')
        clf_coef=clf.coef_
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print(clf_coef)
        print('Intercept: %.3f' % clf.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train_std, y_train_pred),
        mean_squared_error(y_test_std, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_std, y_train_pred),
        r2_score(y_test_std, y_test_pred)))

#Linear regression
slr=LinearRegression()
print("Linear regression(PCT 3MO FWD): ")
train_and_evaluate(slr,X_train_pca,X_test_pca,y_3m_train_std,y_3m_test_std,True)
print("")
print("Linear regression(PCT 6MO FWD): ")
train_and_evaluate(slr,X_train_pca,X_test_pca,y_6m_train_std,y_6m_test_std,True)
print("")
print("Linear regression(PCT 9MO FWD): ")
train_and_evaluate(slr,X_train_pca,X_test_pca,y_9m_train_std,y_9m_test_std,True)
print("")
#svm regression
clf_svr=svm.SVR(kernel="linear")
print("SVM linear regression(PCT 3MO FWD): ")
train_and_evaluate(clf_svr,X_train_pca,X_test_pca,y_3m_train_std,y_3m_test_std,True)
print("")
print("SVM linear regression(PCT 6MO FWD): ")
train_and_evaluate(clf_svr,X_train_pca,X_test_pca,y_6m_train_std,y_6m_test_std,True)
print("")
print("SVM linear regression(PCT 9MO FWD): ")
train_and_evaluate(clf_svr,X_train_pca,X_test_pca,y_9m_train_std,y_9m_test_std,True)
print("")
#svm rbf
rbf=svm.SVR(kernel="rbf",C=1)
print("SVM RBF regression(PCT 3MO FWD):")
train_and_evaluate(rbf,X_train_pca,X_test_pca,y_3m_train_std,y_3m_test_std,False)
print("")
print("SVM RBF regression(PCT 6MO FWD):")
train_and_evaluate(rbf,X_train_pca,X_test_pca,y_6m_train_std,y_6m_test_std,False)
print("")
rbf=svm.SVR(kernel="rbf",C=0.5)
print("SVM RBF regression(PCT 9MO FWD):")
train_and_evaluate(rbf,X_train_pca,X_test_pca,y_9m_train_std,y_9m_test_std,False)
print("")
#svm poly
poly=svm.SVR(kernel="poly")
print("SVM poly regression(PCT 3MO FWD):")
train_and_evaluate(poly,X_train_pca,X_test_pca,y_3m_train_std,y_3m_test_std,False)
print("")
print("SVM poly regression(PCT 6MO FWD):")
train_and_evaluate(poly,X_train_pca,X_test_pca,y_6m_train_std,y_6m_test_std,False)
print("")
print("SVM poly regression(PCT 9MO FWD):")
train_and_evaluate(poly,X_train_pca,X_test_pca,y_9m_train_std,y_9m_test_std,False)
print("")
#SGDRegressor
sgd=SGDRegressor()
print("SGDRegressor(PCT 3MO FWD): ")
train_and_evaluate(sgd,X_train_pca,X_test_pca,y_3m_train_std,y_3m_test_std,True)
print("")
print("SGDRegressor(PCT 6MO FWD): ")
train_and_evaluate(sgd,X_train_pca,X_test_pca,y_6m_train_std,y_6m_test_std,True)
print("")
print("SGDRegressor(PCT 9MO FWD): ")
train_and_evaluate(sgd,X_train_pca,X_test_pca,y_9m_train_std,y_9m_test_std,True)
print("")
#decision tree
tree=DecisionTreeRegressor(max_depth=10)
print("DecisionTreeRegressor(PCT 3MO FWD): ")
train_and_evaluate(tree,X_train_std,X_test_std,y_3m_train_std,y_3m_test_std,False)
print("")
print("DecisionTreeRegressor(PCT 6MO FWD): ")
train_and_evaluate(tree,X_train_std,X_test_std,y_6m_train_std,y_6m_test_std,False)
print("")
print("DecisionTreeRegressor(PCT 9MO FWD): ")
train_and_evaluate(tree,X_train_std,X_test_std,y_9m_train_std,y_9m_test_std,False)
print("")

# LASSO regression: test the best alpha
begin=0.01
end=1
krange=np.arange(begin,end,0.01)
testscores=[]
trainscores=[]
for k in krange:
    lasso = Lasso(alpha=k)
    lasso.fit(X_train_std,y_9m_train_std)
    y_train_pred = lasso.predict(X_train)
    y_test_pred=lasso.predict(X_test)
    score=lasso.score(X_test_std,y_9m_test_std)
    testscores.append(lasso.score(X_test_std,y_9m_test_std))
    trainscores.append(lasso.score(X_train_std,y_9m_train_std))
    
#print best alpha and R^2
maxridge=max(testscores)
max_index=testscores.index(maxridge)
best_alpha=max_index/100+begin
best_alpha
plt.plot(krange, testscores)
plt.xlabel('Alpha values')
plt.ylabel('R^2 of Lasso regression')
plt.title('R^2 with different alpha value using Lasso regression')
plt.xlim([-end/10, end])
plt.show()
print('Lasso:Best alpha=',best_alpha,' R^2=','%.3f' %maxridge)
#alpha 0.02 0.04 0.04
#Lasso
lasso = Lasso(alpha=0.02)
print("Lasso(PCT 3MO FWD): ")
train_and_evaluate(lasso,X_train_std,X_test_std,y_3m_train_std,y_3m_test_std,True)
print("")
lasso = Lasso(alpha=0.04)
print("Lasso(PCT 6MO FWD): ")
train_and_evaluate(lasso,X_train_std,X_test_std,y_6m_train_std,y_6m_test_std,True)
print("")
print("Lasso(PCT 9MO FWD): ")
train_and_evaluate(lasso,X_train_std,X_test_std,y_9m_train_std,y_9m_test_std,True)
print("")

#Randomforest regression
forest=RandomForestRegressor(n_estimators=51,criterion='mse',
                             random_state=1,n_jobs=-1,max_depth=4)
print("RandomForestRegressor(PCT 3MO FWD): ")
train_and_evaluate(forest,X_train_std,X_test_std,y_3m_train_std,y_3m_test_std,False)
print("")
forest=RandomForestRegressor(n_estimators=101,criterion='mse',
                             random_state=1,n_jobs=-1,max_depth=4)
print("RandomForestRegressor(PCT 6MO FWD): ")
train_and_evaluate(forest,X_train_std,X_test_std,y_6m_train_std,y_6m_test_std,False)
print("")
forest=RandomForestRegressor(n_estimators=151,criterion='mse',
                             random_state=1,n_jobs=-1,max_depth=4)
print("RandomForestRegressor(PCT 9MO FWD): ")
train_and_evaluate(forest,X_train_std,X_test_std,y_9m_train_std,y_9m_test_std,False)
print("")

#best n-estimators or max_depth
rangef=list(np.arange(3,11))
params={'max_depth':rangef}
grid = GridSearchCV(estimator=forest,param_grid=params,cv=2)
grid.fit(X_train_std,y_3m_train_std)
y_pred=grid.predict(X_test_std)
results=grid.cv_results_
print('')
print('GridSearch:')
print('Tuned Model Parameters:{}'.format(grid.best_params_))

#Randomforest regression pca
forest=RandomForestRegressor(n_estimators=100,criterion='mse',
                             random_state=1,n_jobs=-1,max_depth=5)
print("RandomForestRegressor(PCT 3MO FWD): ")
train_and_evaluate(forest,X_train_pca,X_test_pca,y_3m_train_std,y_3m_test_std,False)
print("")
print("RandomForestRegressor(PCT 6MO FWD): ")
train_and_evaluate(forest,X_train_pca,X_test_pca,y_6m_train_std,y_6m_test_std,False)
print("")
print("RandomForestRegressor(PCT 9MO FWD): ")
train_and_evaluate(forest,X_train_pca,X_test_pca,y_9m_train_std,y_9m_test_std,False)
print("")

#Bagging to reduce variance__not meaningful for randomforest
bag=BaggingRegressor(base_estimator=forest,n_estimators=100,max_samples=150,
                     max_features=10,bootstrap=True,bootstrap_features=False,
                     n_jobs=-1,random_state=1)
print("BaggingRegressor(PCT 3MO FWD): ")
train_and_evaluate(bag,X_train_std,X_test_std,y_3m_train_std,y_3m_test_std,False)
print("")
print("BaggingRegressor(PCT 6MO FWD): ")
train_and_evaluate(bag,X_train_std,X_test_std,y_6m_train_std,y_6m_test_std,False)
print("")
print("BaggingRegressor(PCT 9MO FWD): ")
train_and_evaluate(bag,X_train_std,X_test_std,y_9m_train_std,y_9m_test_std,False)
print("")

#Final outcomes
#PCT 3MO FWD: Adaboosting for svm rbf 
#PCT 3MO FWD: SVM rbf
rbf=svm.SVR(kernel="rbf",C=1)
params={'C':[0.5,1,5,10]}
grid = GridSearchCV(estimator=rbf,param_grid=params,cv=5)
grid.fit(X_train_std,y_3m_train_std)
y_pred=grid.predict(X_test_std)
results=grid.cv_results_
print('GridSearch(SVM):')
print('Tuned Model Parameters:{}'.format(grid.best_params_))

print("AdaBoostRegressor(PCT 3MO FWD): ")
ada=AdaBoostRegressor(base_estimator=rbf,n_estimators=100,learning_rate=0.1,random_state=1)
train_and_evaluate(ada,X_train_std,X_test_std,y_3m_train_std,y_3m_test_std,False)
print("")
#train_and_evaluate(ada,X_train_std,X_test_std,y_6m_train_std,y_6m_test_std,False)
#train_and_evaluate(ada,X_train_pca,X_test_pca,y_9m_train_std,y_9m_test_std,False)

#PCT 6MO FWD: Randomforest
forest=RandomForestRegressor(n_estimators=180,criterion='mse',
                             random_state=1,n_jobs=-1,max_depth=4)
params={'n_estimators':[110,125,150,180,190]}
grid = GridSearchCV(estimator=forest,param_grid=params,cv=5)
grid.fit(X_train_std,y_6m_train_std)
y_pred=grid.predict(X_test_std)
results=grid.cv_results_
print('GridSearch(RandomForest):')
print('Tuned Model Parameters:{}'.format(grid.best_params_))

print("RandomForestRegressor(PCT 6MO FWD): ")
train_and_evaluate(forest,X_train_std,X_test_std,y_6m_train_std,y_6m_test_std,False)
print("")

#PCT 9MO FWD: Randomforest
forest=RandomForestRegressor(n_estimators=150,criterion='mse',
                             random_state=1,n_jobs=-1,max_depth=4)
params={'n_estimators':[100,120,135,150,180]}
grid = GridSearchCV(estimator=forest,param_grid=params,cv=5)
grid.fit(X_train_std,y_9m_train_std)
y_pred=grid.predict(X_test_std)
results=grid.cv_results_
print('GridSearch(RandomForest):')
print('Tuned Model Parameters:{}'.format(grid.best_params_))

print("RandomForestRegressor(PCT 9MO FWD): ")
train_and_evaluate(forest,X_train_std,X_test_std,y_9m_train_std,y_9m_test_std,False)
print("")
















