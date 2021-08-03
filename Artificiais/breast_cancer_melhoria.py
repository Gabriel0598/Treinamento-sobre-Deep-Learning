# Seção 4 - Exercício

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
    # Camada de entrada com 100 neurônios;
    # Redução de 10 em 10 neurônios;
    # Camada de saída com apenas 1 neurônio;
    # Kernel_initialiazer como orthogonal;
    # Ativação por softmax por se tratarem de camadas muitos profundas;
    classificador = Sequential([
               tf.keras.layers.Dense(units=40, activation = 'softmax', kernel_initializer = 'random_normal', input_dim=50),
               tf.keras.layers.Dropout(0.2),
               # Este dropout zera o valor de 20% dos neurônios da camada de entrada;
               tf.keras.layers.Dense(units=30, activation = 'softmax', kernel_initializer = 'random_normal'),
               tf.keras.layers.Dropout(0.5),
               # Este dropout zera o valor de 50% dos neurônios da segunda camada;
               tf.keras.layers.Dense(units=20, activation = 'softmax', kernel_initializer = 'random_normal'),
               tf.keras.layers.Dropout(0.5),
               # Este dropout zera o valor de 50% dos neurônios da terceira camada;
               tf.keras.layers.Dense(units=10, activation = 'softmax', kernel_initializer = 'random_normal'),
               tf.keras.layers.Dropout(0.5),
               # Este dropout zera o valor de 50% dos neurônios da quarta camada;
               
               # Utilizado apenas um neurônio com função sigmoid na camada de saída (Tendo em vista retornar 0 ou 1);
               tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
    # Sintaxe para criação de Stochastic Gradient Descent:
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    var = tf.Variable(1.0)
    val0 = var.value()
    loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
    # First step is `- learning_rate * grad`  
    step_count = opt.minimize(loss, [var]).numpy()
    val1 = var.value()
    (val0 - val1).numpy()
    0.1
    # On later steps, step-size increases because of momentum
    
    classificador.compile(optimizer = opt, loss = 'MeanSquaredError', metrics = ['binary_accuracy'])
    # Utilizado MSE devido a sua maior precisão para cálculo indivual, binary accuracy também devido a maior precisão;

    return classificador

    classificador = KerasClassifier(build_fn = criarRede, epochs = 200, batch_size = 10)
    resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')

    media = resultados.mean()
    desvio = resultados.std()
# Acusa avisos de não utilização de determinadas variáveis, necessária revisão posterior, algoritmo não roda perfeitamente;
