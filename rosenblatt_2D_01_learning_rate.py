import numpy as np
import matplotlib.pyplot as plt

from graficos_color import perceptron_plot 

#original
#entrada = [ [0.7, 1.3], [2.0, 1.1], [1.0, 1.9],
#            [3.0, 1.0], [1.5, 2.1] ]
#salida = [0,0,0,1,1]
#salida2 = [0,0,0,1,1]
#salida3 = [0,0,0,1,1]


#ave
#entrada = [[0,-4],[0,-3],[0,-2],[0,-1],[0,0],[-1,-4],[-1,-3],[-1,-2],[-1,-1],[-1,0],[5.5,-0.7],[5,-1.2],[6,-2],[4,0.2],[0,3.3], [0.2,3.5],[1,0.5],[3,0.8],[3,1], [0.7, 1.3], [2.0, 1.3],
#            [1.5, 2], [2.5, 2.1], [4,2.5], [3.5,1], [1.5,2.2], [1,2.8], [2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5], [2,4],[3,4],[4,4],[5,4],[6,4],[7,4],[8,4] ]
#salida = [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#salida2 = [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#salida3 = [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

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
salida2=[1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
salida3=[1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida)
Z = np.array(salida2)
Z_0 = np.array(salida3)

# Tamano datos
X_row = X.shape[0]
X_col = X.shape[1]

# cuenta los mal clasificados a través de la funcion verif.
def verif(R, T, S, q=0, X_roww = X_row):
# param entrada dos np.array(salida)
# q parámetro adicional para darle una función particular a verif
    comparison = R == T
    if (q == "sin cambio"):
        equal_arrays = comparison.all()
        return equal_arrays
    # si no hay cambio entre iteraciones subo el learning rate
    if (q == "cuenta"):
        return np.count_nonzero(comparison)
    # devuelve la cantidad de aciertos
    if (q in range(X_roww)):
        fla = False
        for w in range(X_roww-1):
            if (w != q):
                if (R[w] == S[w]):
                    if (T[w] != S[w]):
                        fla = True
        return fla
    # dado un paso particular del perceptron en una fila q
    # vemos si la nueva recta pasa a clasificar mal algun punto que
    # ya había clasificado bien con anterioridad
    # este sería un criterio de achicar el learning rate
    # esto será observado por la variable flig

#inicializo los graficos
grafico = perceptron_plot(X, Y, 0.01)



# Incializo la recta azarosamente
np.random.seed(102192) #mi querida random seed para que las corridas sean reproducibles
W = np.array( np.random.uniform(-0.5, 0.5, size=X_col))
x0 = np.random.uniform(-0.5, 0.5)
grafico.graficar(W, x0, 0, -1,'magenta')

 #recorro siempre TODA la entrada
for fila in range(X_row):
    # calculo el estimulo suma, producto interno
    estimulo = x0*1 + W[0]*X[fila,0] + W[1]*X[fila,1]
    # funcion de activacion, a lo bruto con un if
    if(estimulo>0):
        y = 1
    else:
        y = 0
    Z_0[fila] = y
for fila in range(X_row):
    d = Z_0[fila]
    Z[fila] = d
# clasificación inicial con una recta random       
modificados = X_row - verif(Z, Y, Y, "cuenta")


# Inicializo la iteracion
epoch_limit = 100    # para terminar si no converge
learning_rate = 0.01
ajuste = 1.005
epoch_red_max=2
epoch = 0
epoch_red = 0


while ((modificados > 0) and (epoch < epoch_limit)):
    epoch += 1
    epoch_red += 1
    # recorro siempre TODA la entrada
    # etapa uno: agrandar el learning rate
    if (epoch_red < epoch_red_max):
        for fila in range(X_row):
            # solo si corresponde actualizo  W y x0
            if(Z[fila] != Y[fila]):
                l = learning_rate
                # actualizo W y x0
                flog = True
                flog_limit = 0
                while flog and (flog_limit < 1000):
                    W[0] = W[0] + l * (Y[fila]-Z[fila]) * X[fila,0]
                    W[1] = W[1] + l * (Y[fila]-Z[fila]) * X[fila,1]
                    x0 =   x0   + l * (Y[fila]-Z[fila]) * 1
                    for fila2 in range(X_row):
                        # calculo el estimulo suma, producto interno
                        estimulo = x0*1 + W[0]*X[fila2,0] + W[1]*X[fila2,1]
                        # funcion de activacion, a lo bruto con un if
                        if(estimulo>0):
                            yy = 1
                        else:
                            yy = 0
                        Z[fila2] = yy
                    modificados = X_row - verif(Z, Y, Y, "cuenta")
                    if (modificados == 0):
                        break  
                    flog = verif(Z_0, Z, Y, "sin cambio")
                    # si la nueva recta clasifica los puntos igual que la recta anterior
                    # agrando el learning rate
                    if (flog):
                        l = l*ajuste
                        flog_limit +=1
                grafico.graficar(W, x0, epoch, fila,'magenta') #grafico
            for fila3 in range(X_row):
                d = Z[fila3]
                Z_0[fila3] = d
            modificados = X_row - verif(Z, Y, Y, "cuenta")
            if (modificados == 0):
                break
    if (modificados > 0):    
        # etapa 2: achicar el learning rate
        for fila in range(X_row): 
            # solo si corresponde actualizo  W y x0
            if(Z[fila] != Y[fila]):
                l = learning_rate
                #modificados += 1  # encontre un registro que esta mal clasificado
                # actualizo W y x0
                flug = True
                flig = True
                flug_limit = 0
                while (flig or flug) and (flug_limit < 1000):
                    W[0] = W[0] + l * (Y[fila]-Z[fila]) * X[fila,0]
                    W[1] = W[1] + l * (Y[fila]-Z[fila]) * X[fila,1]
                    x0 =   x0   + l * (Y[fila]-Z[fila]) * 1
                    for fila2 in range(X_row):
                        # calculo el estimulo suma, producto interno
                        estimulo = x0*1 + W[0]*X[fila2,0] + W[1]*X[fila2,1]
                        # funcion de activacion, a lo bruto con un if
                        if(estimulo>0):
                            yy = 1
                        else:
                            yy = 0
                        Z[fila2] = yy
                    modificados = X_row - verif(Z, Y, Y, "cuenta")
                    if (modificados == 0):
                        break   
                    if (modificados > 0):
                        flug = (modificados > (X_row - verif(Z_0, Y, Y, "cuenta")))
                        # otra variable de medida a ver si achicamos el learning rate será
                        # flug que se fija si pasamos a tener o no mayor cantidad de 
                        # modificados
                        # con una combinación de flig y flug decidimos si achicamos el 
                        # learning rate
                        flig = verif(Z_0, Z, Y, fila)
                        if (flig or flug): 
                            # achicamos el learning rate
                            l = l/ajuste 
                            flug_limit += 1
                grafico.graficar(W, x0, epoch, fila,'black') #grafico
            for fila3 in range(X_row):
                d = Z[fila3]
                Z_0[fila3] = d
            modificados = X_row - verif(Z, Y, Y, "cuenta")
            if (modificados == 0):
                break
    if (modificados == 0):
                break            

grafico.graficar(W, x0, epoch, -1, 'red')
print(epoch, W, x0)
