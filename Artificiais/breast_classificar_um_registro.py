# Aula 28

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Dropout # atualizado: tensorflow==2.0.0-beta1

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# Usar parâmetros que o teste indicar que são os melhores:
classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu',
                        kernel_initializer = 'normal', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu',
                        kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

novo = np.array([[15.80, 8.34, 118, 980, 0.10, 0.26, 0.08, 0.134, 0.178,
                 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05,
                 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                 0.84, 158, 0.363]])
"""
É necessário usar dois colchetes para passar os parâmetros,
tendo em vista que o array deve ser armazenado em apenas uma linha;
Valores implementados aqui são aleatórios, trata - se das entradas da rede neural;
"""
previsao = classificador.predict(novo)
# Por apontar uma precisão de 0.99 nota - se que é um tumor maligno;
previsao = (previsao > 0.5)
# Usado 0.5 nesse caso, em cenários em que é necessária uma previsão mais detalhada apontar valores como 0.9 ou 0.95 por exemplo;
