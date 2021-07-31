# Aula 29

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

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

classificador_json = classificador.to_json()
# Trata - se de formato muito utilizado em webservers para tráfego de arquivos pelas rede;
with open('classificador_breast.json', 'w') as json_file:
          json_file.write(classificador_json)
# Operação executa um save em disco de arquivo de json que guarda a parametrização da rede neural;
# Sintaxe 'w' usada para indicar escrita (write) na mesma pasta deste arquivo;
classificador.save_weights('classificador_breast.h5')
# Trata - se de arquivo que armazena os pesos da rede neural em extensão H5;
# Portanto, utilizando estes arquivos json e h5 é possível copiar e implementá-los em nova rede neural;
          