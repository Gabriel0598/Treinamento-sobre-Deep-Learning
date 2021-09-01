import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

"""
Ao ocorrer o problema "module 'keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'",
verificar possível solução no fórum do GitHub: https://github.com/keras-team/keras/issues/14632.
Este problema trata - se de uma incompatibilidade entre a versão do tensorflow na máquina e o compilador,
consegui corrigir este impasse através da atualização do tensorflow através do Anaconda Prompt com o comando:
pip install tensorflow --upate
"""

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede():
    # Construção da estrutura da rede neural:
    classificador = Sequential()
    classificador.add(Dense(units = 5, activation = 'relu', input_dim = 6))
    classificador.add(Dense(units = 3, activation = 'relu'))
    # Duas camadas escondias na rede neural parcial
    classificador.add(Dense(units = 1, activation = 'softmax'))
    # Softmax para problemas de classificação de mais de duas classes
    # Gera uma propabilidade para cada um dos rótulos
    classificador.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criar_rede, epochs = 3000, batch_size = 15)
resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')
# cv = cross validation
media = resultados.mean()
desvio = resultados.std()
# Para verificar overfitting
