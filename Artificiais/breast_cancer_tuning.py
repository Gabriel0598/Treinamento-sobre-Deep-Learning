# Aula 27 - Tuning (Ajuste dos parâmetros)
"""
NOTA - SE QUE PARA REALIZAR A EXECUÇÃO DESTE TESTE SÃO NECESSÁRIAS DIVERSAS HORAS A DEPENDER DO DESEMPENHO DA MÁQUINA.
PORTANTO, RESERVE UM TEMPO SUFICIENTE PARA TAL.
"""

import pandas as pd
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# Pesquisa em grade para encontrar os melhores parâmetros
from tensorflow.keras import backend as k # atualizado: tensorflow==2.0.0-beta1

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons): # atualizado: tensorflow==2.0.0-beta1
# Optimizer para lista de otimizadores;
# Loss function;
# Kernel_initializer, nesse caso como random;
# Activation, nesse caso com ReLU;
# Neurons, para testes dinâmicos de números de neurônios;
    k.clear_session()
    classificador = Sequential([
               tf.keras.layers.Dense(units=neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim=30),
               # Em vez de valores fixos, colocamos os parâmetros a serem aplicados para os testes, no lugar de units, activation e initializer;
               # Input_dim mantém fixo como 30;
               tf.keras.layers.Dropout(0.2),
               # Dropout mantém fixo como 0.2;
               tf.keras.layers.Dense(units=neurons, activation = activation, kernel_initializer = kernel_initializer),
               tf.keras.layers.Dropout(0.2),
               # Camada de saída não recebe alteração devido a ser problema binário:
               tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    # Métrica mantém como binary_accuracy por se tratar de problema de classificação binária;
    return classificador
    """
    Com estes parâmetros o algoritmo realiza um teste geral para determinar quais são:
    as melhores quantidades de neurônios, melhor função de ativação, melhor otimizador e melhor função de perda;
    """

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100], # Em sistemas reais, são apontados testes com milhares ou até milhões de épocas;
              'optimizer': ['adam', 'sgd'], # Stocastic Grad Descent é um pouco inferior ao Adam;
              'loss': ['binary_crossentropy', 'hinge'], # Hinge pode ser outro parâmetro para função de perda;
              'kernel_initializer': ['random_uniform', 'normal'], # Normal 
              'activation': ['relu', 'tanh'], # Tangente hiperbólica traz valores entre -1 e 1;
              'neurons': [16, 8]} # Teste de neurônios com a média do que usamos até o momento, um teste ideal envolve mais neurônios;
# São passados parâmetros num padrão de dicionário;
grid_search = GridSearchCV(estimator = classificador, # Classificador para estimativa;
                           param_grid = parametros, # Grade de parâmetros;
                           scoring = 'accuracy', # Avaliação dos resultado como precisão;
                           cv = 5) # 5 rodadas para teste;
grid_search = grid_search.fit(previsores, classe)
# A grade de pesquisa é fechada pelos parâmetros de previsores e classe
melhores_parametros = grid_search.best_params_
# Esta variável recebe os melhores parâmetros da pesquisa;
melhor_precisao = grid_search.best_score_
# Esta variável recebe os melhores valores da pesquisa;
