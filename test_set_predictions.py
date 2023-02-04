# bibliotecas necessarias
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix,confusion_matrix
from sklearn import svm
import random
import matplotlib.pyplot as plt
import seaborn as sns

# para garantir resultados reproduziveis
rdm = 104
random.seed(rdm)
np.random.seed(rdm)

# le os dados de treino
X_treino = pd.read_csv('data/train/X_train.txt',sep="\s+", header=None)

# le os nomes das features para usar como nome das colunas
with open('data/features.txt') as f:
    feature_names = f.readlines()
feature_names = [item.replace(" ","_").replace("\n","").replace(',','_') for item in feature_names]
X_treino.columns = feature_names

# le dados de treino --> variavel dependente (alvo/target)
y_treino = pd.read_csv('data/train/y_train.txt',sep="\s+", header=None)
y_treino.columns = ['target']

# le os dados de teste
X_teste = pd.read_csv('data/test/X_test.txt',sep="\s+", header=None)

# muda nome das colunas
X_teste.columns = feature_names

# le dados de teste --> variavel dependente (alvo/target)
y_teste = pd.read_csv('data/test/y_test.txt',sep="\s+", header=None)
y_teste.columns = ['target']

# seleciona variaveis com variancia maior que 0.2
var_thresh = VarianceThreshold(threshold=0.2).fit(X_treino, y_treino)
X_treino_selecionado = var_thresh.transform(X_treino)
X_teste_selecionado = var_thresh.transform(X_teste)

# tratamento da variavel 1,2,3,4,5,6 --> 0,1,2,3,4,5
y_treino = y_treino.replace({'target':[1,2,3,4,5,6]},{'target':[0,1,2,3,4,5]})
y_teste = y_teste.replace({'target':[1,2,3,4,5,6]},{'target':[0,1,2,3,4,5]})

dfList = []

# function MCMperf_measure - Calculo de indicadores de performance para target multi-label
def MCMperf_measure(y_actual, y_hat):   # Adaptado da fonte : https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal     TP = 0
   MCM = multilabel_confusion_matrix(y_actual, y_hat,
                              labels=[0, 1, 2, 3, 4, 5])

   aResult = np.empty(shape=[0,13])

   for l in [0, 1, 2, 3, 4, 5]:

      TN = MCM[l][0][0]
      FP = MCM[l][0][1]
      FN = MCM[l][1][0]
      TP = MCM[l][1][1]
      # Sensitivity, hit rate, recall, or true positive rate
      TPR = TP/(TP+FN)
      # Specificity or true negative rate
      TNR = TN/(TN+FP) 
      # Precision or positive predictive value
      PPV = TP/(TP+FP)
      # Negative predictive value
      NPV = TN/(TN+FN)
      # Fall out or false positive rate
      FPR = FP/(FP+TN)
      # False negative rate
      FNR = FN/(TP+FN)
      # False discovery rate
      FDR = FP/(TP+FP)
      # Overall accuracy
      ACC = (TP+TN)/(TP+FP+FN+TN)


      aResult = np.append(aResult, [['Label'+str(l), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC]], axis=0)
      
   return(aResult) 


# modelo SVM
print('Modelo: SVM')
model = 'SVM'
svm_model = svm.SVC(kernel='linear', random_state= rdm)  
svm_model.fit(X_treino,y_treino['target'])
y_pred_svm = svm_model.predict(X_teste)
accuracyTest_svm = accuracy_score(y_teste, y_pred_svm)
print('Acuracia de teste:', accuracyTest_svm)

## insere em dataframe resultados de performance de classificacao por label  -------------------------------------------
aResult = MCMperf_measure(y_teste, y_pred_svm)
print(confusion_matrix(y_teste, y_pred_svm))

for l in range(len(aResult)):
# valores de aResult: label (Label9), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    newRecord = {   'method': model,
                    'label': (aResult[l][0])[-1:],
                    'TN':  aResult[l][1],
                    'FP':  aResult[l][2],
                    'FN':  aResult[l][3],
                    'TP':  aResult[l][4],
                    'TPR': aResult[l][5],
                    'TNR': aResult[l][6],
                    'PPV': aResult[l][7],
                    'NPV': aResult[l][8],
                    'FPR': aResult[l][9],
                    'FNR': aResult[l][10],
                    'FDR': aResult[l][11],
                    'ACC': aResult[l][12]            
                }
    dfList.append(newRecord)


# modelo SVM
print('Modelo: SVM com selecao de features')
model = 'SVM feature selection'
svm_model_select = svm.SVC(kernel='linear', random_state= rdm)  
svm_model_select.fit(X_treino_selecionado,y_treino['target'])
y_pred_svm_select = svm_model_select.predict(X_teste_selecionado)
accuracyTest_svm_select = accuracy_score(y_teste, y_pred_svm_select)
print('Acuracia de teste:', accuracyTest_svm_select)

## insere em dataframe resultados de performance de classificacao por label  -------------------------------------------
aResult = MCMperf_measure(y_teste, y_pred_svm_select)
print(confusion_matrix(y_teste, y_pred_svm_select))

for l in range(len(aResult)):
# valores de aResult: label (Label9), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    newRecord = {   'method': model,
                    'label': (aResult[l][0])[-1:],
                    'TN':  aResult[l][1],
                    'FP':  aResult[l][2],
                    'FN':  aResult[l][3],
                    'TP':  aResult[l][4],
                    'TPR': aResult[l][5],
                    'TNR': aResult[l][6],
                    'PPV': aResult[l][7],
                    'NPV': aResult[l][8],
                    'FPR': aResult[l][9],
                    'FNR': aResult[l][10],
                    'FDR': aResult[l][11],
                    'ACC': aResult[l][12]            
                }
    dfList.append(newRecord)

# modelo Random Forests
print('\nModelo: Random Forest')
model = 'Random Forest'
rdmForest_model = RandomForestClassifier(bootstrap= True, max_depth= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 1000, random_state=rdm)
rdmForest_model.fit(X_treino,y_treino['target'])
y_pred_rdmForest = rdmForest_model.predict(X_teste)
accuracyTest_rdmForest = accuracy_score(y_teste, y_pred_rdmForest)
print('Acuracia de teste:', accuracyTest_rdmForest)
## insere em dataframe resultados de performance de classificacao por label  -------------------------------------------
aResult = MCMperf_measure(y_teste, y_pred_rdmForest)
print(confusion_matrix(y_teste, y_pred_rdmForest))

for l in range(len(aResult)):
# valores de aResult: label (Label9), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    newRecord = {   'method': model,
                    'label': (aResult[l][0])[-1:],
                    'TN':  aResult[l][1],
                    'FP':  aResult[l][2],
                    'FN':  aResult[l][3],
                    'TP':  aResult[l][4],
                    'TPR': aResult[l][5],
                    'TNR': aResult[l][6],
                    'PPV': aResult[l][7],
                    'NPV': aResult[l][8],
                    'FPR': aResult[l][9],
                    'FNR': aResult[l][10],
                    'FDR': aResult[l][11],
                    'ACC': aResult[l][12]            
                }
    dfList.append(newRecord)


# modelo Random Forests
print('\nModelo: Random Forest com selecao de features')
model = 'Random Forest feature selection'
rdmForest_model_select = RandomForestClassifier(bootstrap= True, max_depth= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 1000, random_state=rdm)
rdmForest_model_select.fit(X_treino_selecionado,y_treino['target'])
y_pred_rdnmForest_select = rdmForest_model_select.predict(X_teste_selecionado)
accuracyTest_rdmForest_select = accuracy_score(y_teste, y_pred_rdnmForest_select)
print('Acuracia de teste:', accuracyTest_rdmForest_select)
## insere em dataframe resultados de performance de classificacao por label  -------------------------------------------
aResult = MCMperf_measure(y_teste, y_pred_rdnmForest_select)
print(confusion_matrix(y_teste, y_pred_rdnmForest_select))

for l in range(len(aResult)):
# valores de aResult: label (Label9), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    newRecord = {   'method': model,
                    'label': (aResult[l][0])[-1:],
                    'TN':  aResult[l][1],
                    'FP':  aResult[l][2],
                    'FN':  aResult[l][3],
                    'TP':  aResult[l][4],
                    'TPR': aResult[l][5],
                    'TNR': aResult[l][6],
                    'PPV': aResult[l][7],
                    'NPV': aResult[l][8],
                    'FPR': aResult[l][9],
                    'FNR': aResult[l][10],
                    'FDR': aResult[l][11],
                    'ACC': aResult[l][12]            
                }
    dfList.append(newRecord)


# regressao logistica
print('\nModelo: Regressao Logistica')
model = 'Reg. Logistica'
logReg_model = LogisticRegression (C = 2.0, class_weight = 'balanced', dual = False, fit_intercept = True, intercept_scaling = 1, l1_ratio = None, max_iter = 100, multi_class = 'auto', n_jobs= None, penalty = 'l2', random_state = rdm, solver = 'liblinear', tol = 0.0001, verbose = 0, warm_start = False)
logReg_model.fit(X_treino, y_treino['target'])
y_pred_logReg = logReg_model.predict(X_teste)
accuracyTest_logReg = accuracy_score(y_teste, y_pred_logReg)
print('Acuracia de teste:', accuracyTest_logReg)

## insere em dataframe resultados de performance de classificacao por label  -------------------------------------------
aResult = MCMperf_measure(y_teste, y_pred_logReg)
print(confusion_matrix(y_teste, y_pred_logReg))

for l in range(len(aResult)):
# valores de aResult: label (Label9), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    newRecord = {   'method': model,
                    'label': (aResult[l][0])[-1:],
                    'TN':  aResult[l][1],
                    'FP':  aResult[l][2],
                    'FN':  aResult[l][3],
                    'TP':  aResult[l][4],
                    'TPR': aResult[l][5],
                    'TNR': aResult[l][6],
                    'PPV': aResult[l][7],
                    'NPV': aResult[l][8],
                    'FPR': aResult[l][9],
                    'FNR': aResult[l][10],
                    'FDR': aResult[l][11],
                    'ACC': aResult[l][12]            
                }
    dfList.append(newRecord)


# regressao logistica
print('\nModelo: Regressao Logistica com selecao de features')
model = 'Reg. Logistica feature selection'
logReg_model_select = LogisticRegression (C = 2.0, class_weight = 'balanced', dual = False, fit_intercept = True, intercept_scaling = 1, l1_ratio = None, max_iter = 100, multi_class = 'auto', n_jobs= None, penalty = 'l2', random_state = rdm, solver = 'liblinear', tol = 0.0001, verbose = 0, warm_start = False)
logReg_model_select.fit(X_treino_selecionado, y_treino['target'])
y_pred_logReg_select = logReg_model_select.predict(X_teste_selecionado)
accuracyTest_logReg_select = accuracy_score(y_teste, y_pred_logReg_select)
print('Acuracia de teste:', accuracyTest_logReg_select)

## insere em dataframe resultados de performance de classificacao por label  -------------------------------------------
aResult = MCMperf_measure(y_teste, y_pred_logReg_select)
print(confusion_matrix(y_teste, y_pred_logReg_select))

for l in range(len(aResult)):
# valores de aResult: label (Label9), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    newRecord = {   'method': model,
                    'label': (aResult[l][0])[-1:],
                    'TN':  aResult[l][1],
                    'FP':  aResult[l][2],
                    'FN':  aResult[l][3],
                    'TP':  aResult[l][4],
                    'TPR': aResult[l][5],
                    'TNR': aResult[l][6],
                    'PPV': aResult[l][7],
                    'NPV': aResult[l][8],
                    'FPR': aResult[l][9],
                    'FNR': aResult[l][10],
                    'FDR': aResult[l][11],
                    'ACC': aResult[l][12]            
                }
    dfList.append(newRecord)

# knn
print('\nModelo: KNN')
model = 'KNN'
knn_model = KNeighborsClassifier(n_neighbors=11)
knn_model.fit(X_treino,y_treino['target'])
y_pred_knn = knn_model.predict(X_teste)
accuracyTest_knn = accuracy_score(y_teste, y_pred_knn)
print('Acuracia de teste:', accuracyTest_knn)

## insere em dataframe resultados de performance de classificacao por label  -------------------------------------------
aResult = MCMperf_measure(y_teste, y_pred_knn)
print(confusion_matrix(y_teste, y_pred_knn))

for l in range(len(aResult)):
# valores de aResult: label (Label9), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    newRecord = {   'method': model,
                    'label': (aResult[l][0])[-1:],
                    'TN':  aResult[l][1],
                    'FP':  aResult[l][2],
                    'FN':  aResult[l][3],
                    'TP':  aResult[l][4],
                    'TPR': aResult[l][5],
                    'TNR': aResult[l][6],
                    'PPV': aResult[l][7],
                    'NPV': aResult[l][8],
                    'FPR': aResult[l][9],
                    'FNR': aResult[l][10],
                    'FDR': aResult[l][11],
                    'ACC': aResult[l][12]            
                }
    dfList.append(newRecord)


# knn
print('\nModelo: KNN com selecao de features')
model = 'KNN feature selection'
knn_model_select = KNeighborsClassifier(n_neighbors=11)
knn_model_select.fit(X_treino_selecionado,y_treino['target'])
y_pred_knn_select = knn_model_select.predict(X_teste_selecionado)
accuracyTest_knn_select = accuracy_score(y_teste, y_pred_knn_select)
print('Acuracia de teste:', accuracyTest_knn_select)

## insere em dataframe resultados de performance de classificacao por label  -------------------------------------------
aResult = MCMperf_measure(y_teste, y_pred_knn_select)
print(confusion_matrix(y_teste, y_pred_knn_select))

for l in range(len(aResult)):
# valores de aResult: label (Label9), TN, FP, FN, TP, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    newRecord = {   'method': model,
                    'label': (aResult[l][0])[-1:],
                    'TN':  aResult[l][1],
                    'FP':  aResult[l][2],
                    'FN':  aResult[l][3],
                    'TP':  aResult[l][4],
                    'TPR': aResult[l][5],
                    'TNR': aResult[l][6],
                    'PPV': aResult[l][7],
                    'NPV': aResult[l][8],
                    'FPR': aResult[l][9],
                    'FNR': aResult[l][10],
                    'FDR': aResult[l][11],
                    'ACC': aResult[l][12]            
                }
    dfList.append(newRecord)


# monta uma dataframe com as metricas por label
dfMCM = pd.DataFrame.from_records(dfList)
dfMCM = dfMCM.apply(pd.to_numeric, errors='ignore')

# salva os resultados
dfMCM[['method','label','TPR','PPV','ACC']].to_csv('metrica_labels.csv', index=False)

# troca 'feature selection' por * para melhorar a leitura do grafico
dfMCM['method'] = dfMCM['method'].replace('feature selection', '*', regex=True)

#precision
print('Salvar grafico precision')
plt.clf()
plt.xticks(rotation=90)
plt.ylim(0.70, 1.05)
plt.grid() 
ax = sns.barplot(data=dfMCM, x="method", y="PPV", hue="label")
plt.ylabel("Precision")
plt.title("Precision para cada label, para cada modelo")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

fig = ax.get_figure()
fig.savefig("figures/precison_label.png", bbox_inches="tight") 


# recall
print('Salvar grafico recall')
plt.clf()
plt.xticks(rotation=90)
plt.ylim(0.70, 1.05)
plt.grid() 
ax = sns.barplot(data=dfMCM, x="method", y="TPR", hue="label")
plt.ylabel("Recall")
plt.title("Recall para cada label, para cada modelo")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

fig = ax.get_figure()
fig.savefig("figures/recall_label.png", bbox_inches="tight") 


# accuracy
print('Salvar grafico acuracia')
plt.clf()
plt.xticks(rotation=90)
plt.ylim(0.90, 1.02)
plt.grid() 
ax = sns.barplot(data=dfMCM, x="method", y="ACC", hue="label")
plt.ylabel("Acuracia")
plt.title("Acuracia para cada label, para cada modelo")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

fig = ax.get_figure()
fig.savefig("figures/accuracy_label.png", bbox_inches="tight") 


