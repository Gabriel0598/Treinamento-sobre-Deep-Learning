# Aulas 23 a 26

import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# Importada bilioteca para classificação
from sklearn.model_selection import cross_val_score
# Importada biblioteca do modelo de aprendizagem para treinamento de validação cruzada;
from tensorflow.keras.layers import Dense, Dropout
# Importados os layers Dense e Dropout conforme aula, porém por algum outro parâmetro que previamente os importa, acusa aviso que não estão sendo usados;

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(): # atualizado: tensorflow==2.0.0-beta1
    k.clear_session()
    classificador = Sequential([
               tf.keras.layers.Dense(units=16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim=30),
               tf.keras.layers.Dropout(0.2),
               # Este dropout zera o valor de 20% dos neurônios da camada de entrada;
               tf.keras.layers.Dense(units=16, activation = 'relu', kernel_initializer = 'random_uniform'),
               tf.keras.layers.Dropout(0.2),
               # Este dropout zera o valor de 20% dos neurônios da segunda camada;
               tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
    otimizador = tf.keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    # Usado mesmo parâmetro de cálculo que a rede binária simples

    return classificador

classificador = KerasClassifier(build_fn = criarRede, epochs = 100, batch_size = 10)
# build_fn é o parâmetro para criação da rede, tanto que atribui sua operação na variável criarRede;
# São passadas 100 épocas como parâmetro de cálculo;
# De 10 em 10 são feitos os testes;
resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')
# Nesse novo treinamento é usado um padrão de teste e treinamento de validação cruzada;
# K-fold cross validation;
# Estimator recebe toda a classificação gerada nas épocas;
# Dessa forma a porção de testes recebe uma parte e as outras nove são usadas para treinamento;
# Assim após as 10 rodadas todas as partes foram treinadas e receberam testes;
# Os demais abaixo são parâmetros para esta validação:
# X precisa ser maiúsculo como parâmetro
# y precisa ser minúsculo como parâmetro
# cv são as rodadas de teste, nesse caso 10
# scoring recebe padrão accuracy

media = resultados.mean()
# Cálculo da média = 0.85 (85% de média de acerto nesta base de dados sem uso de dropout)
# 0.86 (86% de média de acerto com uso de dropout)
desvio = resultados.std()
# Cálculo do desvio padrão para saber quantos valores estão variando em relação a média
# Desvio padrão em 0.06 em relação a média (Sem uso de dropout)/ 0.062 de desvio (Com uso de dropout): Pequena redução do desvio;
# Maior resultado foi de 0.98 sem uso de dropout;
# Maior resultado foi de 0.96 com uso de dropout, ou seja, uma queda no alcance;
# Menor resultado foi de 0.75 sem uso de dropout;
# Menor resultado foi de 0.754 com uso de dropout, ou seja, um ajuste pequeno no menor alcance;
# Overfitting acontece se o desvio padrão for alto, nesse caso é baixo;
# Pelo overfitting, a base teria dificuldade de previsão por receber um padrão diferente;
# Portanto, usando o parâmetro dropout, há uma melhoria na precisão do treinamento, mesmo que pequena;
# Isso evita o acontecimento de overfitting ou underfitting;