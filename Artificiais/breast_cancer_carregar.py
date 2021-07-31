# Aula 30

import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
# Função de abertura do arquivo json que já tem a parametrização necessária;
# 'r' para realizar a leitura desse arquivo implementado;
# importado modelo do formato json
estrutura_rede = arquivo.read()
# Abrir estrutura da rede
arquivo.close()
# Fecha arquivo para liberar memória

classificador = model_from_json(estrutura_rede)
# classificador receberá de forma simples a estrutura da rede previamente criada que foi salva no arquivo json;
classificador.load_weights('classificador_breast.h5')
# Peso para treinamento será carregado do que foi criado previamente;

novo = np.array([[15.80, 8.34, 118, 980, 0.10, 0.26, 0.08, 0.134, 0.178,
                 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05,
                 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                 0.84, 158, 0.363]])
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')
classificador.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['binary_accuracy'])
# Parâmetros que previamente foram apontados como os melhores para esta situação;
resultado = classificador.evaluate(previsores, classe)
# Resultado retorna:
# Loss function = 0.16
# Accuracy = 0.94
# Portanto, é possível usar grandes bases de dados através deste método para fazer esta avaliação;
