# bibliotecas necessarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importa dados de treino
df = pd.read_csv('data/train/y_train.txt',sep="\s+", header=None)

# renomeia a coluna
df.columns = ['target']

# transforma labels numericas nas labels das acoes
col         = 'target'
conditions  = [ df[col] == 1, df[col] == 2, df[col] == 3, df[col] == 4, df[col] == 5, df[col] == 6]
choices     = [ 'WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
df["target"] = np.select(conditions, choices, default=np.nan)

# conta ocorrencias por acao
df2 = df['target'].value_counts().rename_axis('target').reset_index(name = 'counts')

# define os valores para o gráfico
labels = df2.target
values = df2.counts

# faz o grafico
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.pie(values, labels = labels, autopct= '%1.2f%%')
ax.set(aspect="equal", title='Distribuição das classes no dataset de treino')

# salva a figura
fig.savefig("figures/class_distribution_plot.png", bbox_inches='tight')


