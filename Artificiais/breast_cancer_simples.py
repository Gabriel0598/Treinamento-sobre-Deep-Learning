# Aulas 17 a 22
# Sistema de identificação de câncer
# Saída 0 - Tumor benigno
# Saída 1 - Tumor maligno
# Documentação do Keras detalha comandos de inicialização, funções de ativação, etc...

import pandas as pd
# Biblioteca para manipulação e análise de dados

previsores = pd.read_csv('entradas_breast.csv')
# Variável previsores armazena todos os dados do arquivo csv citado;

classe = pd.read_csv('saidas_breast.csv')
# Classe para realizar previsão

# Necessária base de dados de treinamento e outra base de dados de teste
from sklearn.model_selection import train_test_split
# Scikit-learn é biblioteca para aprendizado de máquina;
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)
# 75% da base de dados sendo usada para treinamento
# 25% da base de dados sendo usada para testes
# A base desses conjuntos é usada para fazer a avaliação do desempenho da rede neural
# Trata - se de problema de classificação binária

import tensorflow as tf # Atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras import backend as k # Atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # Atualizado: tensorflow==2.0.0-beta1


# Variável classificador recebe o modelo sequencial de camadas

# Sendo aplicados 16 neurônios através de units;
# (Entradas + Saídas) / 2 = Números de neurônios das camadas ocultas, arrendondar quando necessário;
# (30 + 1) / 2
# Entradas é definida como 30 devido ao tamanho dos previsores
# Só pode haver uma saída sendo 0 ou 1 nesse caso;
# Camada de entrada está implícita;
# Ativação é feita através da função relu;
# Kernel initializer define quais serão os pesos de entrada, nesse caso aleatórios 'random_unit';
# Input_dim define quantos elementos há na camadas de entrada, nesse caso 30;
# Aplicados 16 neurônios por duas vezes nas camadas ocultas, ou seja, 32 neurônios totais na camada oculta;

# Estrutura da rede neural:
classificador = Sequential([ # atualizado: tensorflow==2.0.0-beta1
               tf.keras.layers.Dense(units=16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim=30),
               tf.keras.layers.Dense(units=16, activation = 'relu', kernel_initializer = 'random_uniform'),
               tf.keras.layers.Dense(units=1, activation = 'sigmoid')])

# Esta linha anterior definiu a camada de saída;
# Nesse caso recebe apenas um neurônio, sendo resultado próximo de 0 ou 1;
# Ativação feita por sigmoid pois retorna 0 ou 1, como sendo a probabilidade de um eventos acontecer;

otimizador = tf.keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5) # Atualizado: tensorflow==2.0.0-beta1
# Otimizador é usado para aumentar a precisão das previsões;
# Learning rate é aplicado para chegar no máximo global;
# Decay é decaimento da taxa de aprendizagem;
# Lr e decay devem possuir valores baixos;
# Clipvalue prende o valor máximo ou mínimo de preso, congela para evitar dispersão no gradiente;
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
# Otimizador faz ajuste dos pesos
# Método de descida do gradiente estocástico por Adam
# Loss para perda;
# Binary_crossentropy para apenas duas classe/ Categorical_crossentropy para várias classes
# Métrica para avaliação é binary_accuracy (precisão binária)

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)
# Encaixar (fit) parâmetros no classificador;
# Épocas passa parâmetros para aumentar a taxa de aprendizagem pelo tempo;
# Batch size define o número de amostras a serem treinadas por comandos, para não extrapolar um alto uso de memória;
# Nesse caso treinaremos de 10 em 10 amostras até atingir as 100 totais;
# A cada 10 ele vai atualizando os erros;

pesos0 = classificador.layers[0].get_weights()
# Pesos0 recebe o peso da primeira camada(0);
print(pesos0)
# Mostra no console os pesos efetivamente gerados que foram aplicados a este treinamento;
print(len(pesos0))
pesos1 = classificador.layers[1].get_weights()
# Próxima variável recebe peso da próxima camada;
pesos2 = classificador.layers[2].get_weights()
# Pesos2 pula para última camada;

# Unidades de BIAS adiciona mais um neurônio na camada para aplicar ligação com o neurônio da próxima camada;

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
# Previsões como boolean para teste de probabilidade
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
# Parâmetros anteriores aumentam a precisão do teste
# Matriz de confusão mostra quantos registros estão sendo aplicados como 0 ou 1

resultado = classificador.evaluate(previsores_teste, classe_teste)
# Classifica os previsores já submetendo registros de classes
