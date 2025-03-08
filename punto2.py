import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


w = 0.8  
b = 0.5

x = np.linspace(-10, 10, 100)

z = w * x + b
output = sigmoid(z)


plt.figure(figsize=(10, 6))
plt.plot(x, output, label='Salida del perceptrón', color='blue')
plt.title('Salida del Perceptrón con Función Sigmoide')
plt.xlabel('Entrada (x)')
plt.ylabel('Salida (ŷ)')
plt.grid(True)
plt.legend()
plt.show()