# Curso Udemy sobre Deep Learning com Python
# Estudo de redes neurais
# Geração de algoritmos de IA

import numpy as np
# Biblioteca numpy é usada para realizar cálculos em arrays multidimensionais
# np é nome dado para quando ela for chamada;

# Transfer Function
# def é usado para implementar função (Realizam tarafas executadas várias vezes através desses blocos de código);

# Funções de ativação aplicadas abaixo:

# Função de ativação para problema linearmente separável:
def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

# Função sidmoid é definida por y = 1 / (1 + ex)
# variável exp é exponencial
# Usada para problemas de classificação binária
def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

# Retorna valores entre -1 e 1 (Função tangente hiperbólica)
def tahnFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))
    # Cálculo através da função Hyperbolic Tanget: Y = ex - (e-x) / ex + (e-x)

# Função ReLU é muito usado para redes neurais profundas (muitas camadas)/ redes neurais convolucionais;
def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0

# Linear function pode ser usado para problema de regressão (Traz o valor preciso que foi implementado)/ Ela não realiza tratamento dos dados;
# Passa o valor pra frente;
def linearFunction(soma):
    return soma


# Softmax function retorna probabilidades em problemas com mais de duas classes
# Recebe diversos valores, como se fosse um vetor
# ex = exponencial
# np traz a aplicação de numpy para realizar cálculo exponencial
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()
    # Sum é o somatório do exponencial;

# Valores aplicados são para teste:
teste = stepFunction(-1)
teste = sigmoidFunction(0.358)
# Como exemplo 0.358 é valor de soma para geração ativação através de sigmoid;
teste = tahnFunction(-0.358)
# Retorna valor negativo;
teste = reluFunction(0.358)
# Se for digitado negativo, como exemplo -0.358 retornará valor 0; Se for superior a zero retornará o próprio valor;
teste = linearFunction(-0.358)
valores = [7.0, 2.0, 1.3]
# Variável valores trata - se de vetor que vieram da camada de saída
print(softmaxFunction(valores))
# Imprime a função softmax aplicada a valores

# Exercícios:

"""    
Neste exercício é passado valor da função soma previamente cálculado
em Heavside Step Function como 2.1;
"""

soma2 = 2.1

def sigmoidFunction2(soma2):
    return 1 / (1 + np.exp(-soma2))

def tahnFunction2(soma2):
    return (np.exp(soma2) - np.exp(-soma2)) / (np.exp(soma2) + np.exp(-soma2))

def reluFunction2(soma2):
    if soma2 >= 0:
        return soma2
    return 0

def linearFunction2(soma2):
    return soma2

# Teste dos valores usando valor de soma2 como parâmetros
teste2 = sigmoidFunction2(soma2)
teste2 = tahnFunction2(soma2)
teste2 = reluFunction2(soma2)
teste2 = linearFunction2(soma2)

# Resposta correta seria 0.89 0.97 2.1 2.1
# Possivelmente devido a erro de cálculo o retorno foi incorreto