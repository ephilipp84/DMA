import numpy as np
import matplotlib.pyplot as plt

from graficos_color import perceptron_plot 

#original
#entrada = [ [0.7, 1.3], [2.0, 1.1], [1.0, 1.9],
#            [3.0, 1.0], [1.5, 2.1] ]
#salida = [0,0,0,1,1]

#ave
#entrada = [[0,-4],[0,-3],[0,-2],[0,-1],[0,0],[-1,-4],[-1,-3],[-1,-2],[-1,-1],[-1,0],[5.5,-0.7],[5,-1.2],[6,-2],[4,0.2],[0,3.3], [0.2,3.5],[1,0.5],[3,0.8],[3,1], [0.7, 1.3], [2.0, 1.3],
#            [1.5, 2], [2.5, 2.1], [4,2.5], [3.5,1], [1.5,2.2], [1,2.8], [2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5], [2,4],[3,4],[4,4],[5,4],[6,4],[7,4],[8,4] ]
#salida = [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

#signo division
E=0.1
entrada = [[0,2],[0,-2],[-5,0],[-4.9,E],[-4.8,0],[-4.7,E],[-4.6,0],[-4.5,E],[-4.4,0],[-4.3,E],[-4.2,0],[-4.1,E],[-4,0],
           [-3.9,E],[-3.8,0],[-3.7,E],[-3.6,0],[-3.5,E],[-3.4,0],[-3.3,E],[-3.2,0],[-3.1,E],[-3,0],
           [-2.9,E],[-2.8,0],[-2.7,E],[-2.6,0],[-2.5,E],[-2.4,0],[-2.3,E],[-2.2,0],[-2.1,E],[-2,0],
           [-1.9,E],[-1.8,0],[-1.7,E],[-1.6,0],[-1.5,E],[-1.4,0],[-1.3,E],[-1.2,0],[-1.1,E],[-1,0],
           [-0.9,E],[-0.8,0],[-0.7,E],[-0.6,0],[-0.5,E],[-0.4,0],[-0.3,E],[-0.2,0],[-0.1,E],[0,0],
           [0.1,E],[0.2,0],[0.3,E],[0.4,0],[0.5,E],[0.6,0],[0.7,E],[0.8,0],[0.9,E],[1,0],
           [1.1,E],[1.2,0],[1.3,E],[1.4,0],[1.5,E],[1.6,0],[1.7,E],[1.8,0],[1.9,E],[2,0],
           [2.1,E],[2.2,0],[2.3,E],[2.4,0],[2.5,E],[2.6,0],[2.7,E],[2.8,0],[2.9,E],[3,0],
           [3.1,E],[3.2,0],[3.3,E],[3.4,0],[3.5,E],[3.6,0],[3.7,E],[3.8,0],[3.9,E],[4,0],
           [4.1,E],[4.2,0],[4.3,E],[4.4,0],[4.5,E],[4.6,0],[4.7,E],[4.8,0],[4.9,E],[5,0]]

salida=[1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]


# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida)

#incializo los graficos
grafico = perceptron_plot(X, Y, 0.01)

# Tamano datos
X_row = X.shape[0]
X_col = X.shape[1]

# Incializo la recta azarosamente
np.random.seed(102192) #mi querida random seed para que las corridas sean reproducibles
W = np.array( np.random.uniform(-0.5, 0.5, size=X_col))
x0 = np.random.uniform(-0.5, 0.5)


# Inicializo la iteracion
epoch_limit = 100    # para terminar si no converge
learning_rate = 0.01
modificados = 1      # lo debo poner algo distinto a 0 la primera vez
epoch = 0

while (modificados and (epoch < epoch_limit)):
    epoch += 1
    modificados = 0  #lo seteo en cero

    #recorro siempre TODA la entrada
    for fila in range(X_row):
        # calculo el estimulo suma, producto interno
        estimulo = x0*1 + W[0]*X[fila,0] + W[1]*X[fila,1]

        # funcion de activacion, a lo bruto con un if
        if(estimulo>0):
            y = 1
        else:
            y = 0

        # solo si corresponde actualizo  W y x0
        if(y != Y[fila]):
            modificados += 1  # encontre un registro que esta mal clasificado
            # actualizo W y x0
            W[0] = W[0] + learning_rate * (Y[fila]-y) * X[fila,0]
            W[1] = W[1] + learning_rate * (Y[fila]-y) * X[fila,1]
            x0 =   x0   + learning_rate * (Y[fila]-y) * 1
            grafico.graficar(W, x0, epoch, fila,'red') #grafico

        

grafico.graficar(W, x0, epoch, -1,'red')
print(epoch, W, x0)
