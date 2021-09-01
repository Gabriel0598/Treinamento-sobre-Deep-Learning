import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
# Sequential se refere a camada sequencial (Uma seguinte a outra)
# Dense se refere a camada densa (Fully conected)

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
# iloc trata - se de função utilizada para divisão
# Values para converter no formato numpy
classe = base.iloc[:, 4].values
# : Indica todas seleção de todas as linhas;

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
# Converter de categórico para númerico, como um cast dos dados;
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 1

from sklearn.model_selection import train_test_split
# Importação de base de dados já incluindo treinamento e teste;
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

# Construção da estrutura da rede neural:
classificador = Sequential()
classificador.add(Dense(units = 5, activation = 'sigmoid', input_dim = 6))
# Dropout para zerar valores (0.2 é ideal na camada de entrada);
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation = 'sigmoid'))
# (0.2) de Dropout ideal nas camadas ocultas;
classificador.add(Dropout(0.2))
# Duas camadas escondias na rede neural parcial
classificador.add(Dense(units = 1, activation = 'sigmoid'))
# Softmax para problemas de classificação de mais de duas classes
# Gera uma propabilidade para cada um dos rótulos
classificador.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
# Compilador geral
# CategoricalAccuracy calcula a frequência com que as previsões correspondem aos rótulos
# kullback_leibler_divergence para classificação de duas classes
# Rede neural construída
# Realizando os testes e a avaliação nela mesma

# Treinamento:
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 15, epochs = 3000)
# Converter de categórico (string) para númerico;

resultado = classificador.evaluate(previsores_teste, classe_teste)
# Teste na base de dados de treinamento

# Visualização da matriz de confusão:
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2)