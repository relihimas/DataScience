import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv()
classe = pd.read_csv()


#divindo os datasets em 25%
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()

# camada oculta
# iniciar os neuronios
# units = (30 + 1)/2 = 16 neuronios
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))

# adição de mais uma camada oculta (parte de otimização)
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))

# camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

# montando um otimizador (parte de otimização)
otimizador = keras.optimizers.Adam(lr = 0.001,decay = 0.0001, clipvalue = 0.5)

# descida do gradiente
# OLD classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)

# camada intermediaria 1
pesos0 = classificador.layers[0].get_weights()
print(pesos0)

# camada intermediaria 2 
pesos1 = classificador.layers[1].get_weights()
print(pesos1)

# camada final
pesos2 = classificador.layers[2].get_weights()
print(pesos2)

previsoes = classificador.predict(previsores_teste)

previsoes_boo = (previsoes > 0.5)
precisao = accuracy_score(classe_teste, previsoes_boo)

print(round((precisao * 100),2))

matriz = confusion_matrix(classe_teste, previsoes_boo)
print(matriz)

resultado = classificador.evaluate(previsores_teste, classe_teste)
