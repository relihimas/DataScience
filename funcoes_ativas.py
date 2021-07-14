import numpy as np


# A função soma é a o valor do neurônio de entrada multiplicado pelo peso
# Ex: neuronio 1 = 0 / neuronio 2 = 1 / peso 1 = -0.424 / peso 2 = 0.358
# Função soma = 0 * (-0.424) + 1 * (0.358) = 0.358

'''Função de ativação - transfer function'''
# Realiza a transferência de valores de uma parte da rede neural para a outra
# que será usado para determinar a saída de um neuronio

# Step Fuction
def stepFunction(soma):
    if soma >= 1:
        print(1)
        return 1
    else:
        print(0)
        return 0

# Usamos a função Sigmoide para determinar a função de ativação para o neurônio
# Essa função determinará, através de um cálculo não linear,

def sigmoideFunciton(soma):
    sig = 1 / (1 + np.exp(-soma))
    print(sig)
    return sig

# Hyperbolic Tanget (retorna valores entre -1 e 1 - usado para classificação de duas classes)

def tahnFunction(soma):
    tahn = (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))
    print(tahn)

# ReLU - rectified linear unites
# retorna zero ou valores maiores que zero - não há valor máximo

def reLu(soma):
    if soma >= 0:
        print(soma)
    else:
        print(0)

# Linear Function
# Muito usada para problemas de regressão

def linearFunction(soma):
    print(soma)

# SoftMax (UMA DAS MAIS IMPORTANTES)
# Retorna probabilidades com problemas de mais classes

def softmaxFunction(x):
    ex = np.exp(x)
    print(ex/ex.sum())



teste = stepFunction(30)
teste2 = sigmoideFunciton(2.1)
teste3 = tahnFunction(2.1)
teste4 = reLu(2.1)
teste5 = linearFunction(2.1)

valores = [5.0, 2.0, 1.3]

teste6 = softmaxFunction(valores)

