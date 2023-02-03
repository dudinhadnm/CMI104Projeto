# bibliotecas usadas
import pandas as pd
import numpy as np

# importa dados de treino
df = pd.read_csv('data/train/X_train.txt',sep="\s+", header=None)

# le os nomes das features para usar como nome das colunas
with open('data/features.txt') as f:
    feature_names = f.readlines()

feature_names = [item.replace(" ","_").replace("\n","").replace(',','_') for item in feature_names]
df.columns = feature_names

# calcula variancia de cada feature
var_columns = df.var()

dfv = var_columns.to_frame(name="variance")
dfv = dfv.reset_index(names = 'feature')

# ordena em ordem decrescente de variancia
dfv = dfv.sort_values(by=['variance'], ascending=False)

# plota o gráfico
plot = dfv.plot(kind='bar',y='variance',x='feature',color='r', xticks = [item*50 for item in list(range(12))], xlabel = 'Features (ordered by variance)',ylabel = 'Variance', legend = False)

# salva a figura em png
fig = plot.get_figure()
fig.savefig("figures/variance_plot.png")


