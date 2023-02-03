# bibliotecas necessarias
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import random

# para garantir resultados reproduziveis
rdm = 104
random.seed(rdm)
np.random.seed(rdm)

# le os dados
X_treino = pd.read_csv('data/train/X_train.txt',sep="\s+", header=None)

# le os nomes das features para usar como nome das colunas
with open('data/features.txt') as f:
    feature_names = f.readlines()
feature_names = [item.replace(" ","_").replace("\n","").replace(',','_') for item in feature_names]
X_treino.columns = feature_names

# le dados de treino --> variavel dependente (alvo/target)
y_treino = pd.read_csv('data/train/y_train.txt',sep="\s+", header=None)
y_treino.columns = ['target']

# metrica a ser avaliadas na validacao cruzada 
scoring = 'accuracy'

X_treino_selecionado = VarianceThreshold(threshold=0.2).fit_transform(X_treino, y_treino)

# knn
print('\nModelo: KNN')
knn_model = KNeighborsClassifier(n_neighbors=11)
scores_knn = cross_val_score(knn_model, X_treino, y_treino['target'], cv=5, scoring=scoring)
print('Acuracia de validacao:', scores_knn.mean())

# knn
print('\nModelo: KNN com selecao de features')
knn_model = KNeighborsClassifier(n_neighbors=11)
scores_knn = cross_val_score(knn_model, X_treino_selecionado, y_treino['target'], cv=5, scoring=scoring)
print('Acuracia de validacao:', scores_knn.mean())


# regressao logistica
print('\nModelo: Regressao Logistica')
logReg_model = LogisticRegression (C = 2.0, class_weight = 'balanced', dual = False, fit_intercept = True, intercept_scaling = 1, l1_ratio = None, max_iter = 100, multi_class = 'auto', n_jobs= None, penalty = 'l2', random_state = rdm, solver = 'liblinear', tol = 0.0001, verbose = 0, warm_start = False)
scores_logReg = cross_val_score(logReg_model, X_treino, y_treino['target'], cv=5, scoring=scoring)
print('Acuracia de validacao:', scores_logReg.mean())


# regressao logistica
print('\nModelo: Regressao Logistica com selecao de features')
logReg_model = LogisticRegression (C = 2.0, class_weight = 'balanced', dual = False, fit_intercept = True, intercept_scaling = 1, l1_ratio = None, max_iter = 100, multi_class = 'auto', n_jobs= None, penalty = 'l2', random_state = rdm, solver = 'liblinear', tol = 0.0001, verbose = 0, warm_start = False)
scores_logReg = cross_val_score(logReg_model, X_treino_selecionado, y_treino['target'], cv=5, scoring=scoring)
print('Acuracia de validacao:', scores_logReg.mean())


# modelo Random Forests
print('\nModelo: Random Forest')
rdmForest_model = RandomForestClassifier(bootstrap= True, max_depth= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 1000, random_state=rdm)
scores_rdmForest = cross_val_score(rdmForest_model, X_treino, y_treino['target'], cv=5, scoring=scoring)
print('Acuracia de validacao:', scores_rdmForest.mean())


# modelo Random Forests
print('\nModelo: Random Forest com selecao de features')
rdmForest_model = RandomForestClassifier(bootstrap= True, max_depth= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 1000, random_state=rdm)
scores_rdmForest = cross_val_score(rdmForest_model, X_treino_selecionado, y_treino['target'], cv=5, scoring=scoring)
print('Acuracia de validacao:', scores_rdmForest.mean())


# modelo SVM
print('Modelo: SVM')
svm_model = svm.SVC(kernel='linear', random_state= rdm)  
scores_svm = cross_val_score(svm_model, X_treino, y_treino['target'], cv=5, scoring=scoring)
print('Acuracia de validacao:', scores_svm.mean())


# modelo SVM
print('Modelo: SVM com selecao de features')
svm_model = svm.SVC(kernel='linear', random_state= rdm)  
scores_svm = cross_val_score(svm_model, X_treino_selecionado, y_treino['target'], cv=5, scoring=scoring)
print('Acuracia de validacao:', scores_svm.mean())
