import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


w = 0.8  # Peso
b = 0.5  # Bias

x_values = np.linspace(-5, 5, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def perceptron_output(x):
    return sigmoid(w * x + b)


def maclaurin_sigmoid(x, N):
    t = w * x + b 
    maclaurin_exp = sum(((-t)**n / math.factorial(n)) for n in range(N))
    return 1 / (1 + maclaurin_exp)


y_sigmoid = np.array([perceptron_output(x) for x in x_values])

N_values = [3, 6, 9]
y_maclaurin = {N: np.array([maclaurin_sigmoid(x, N) for x in x_values]) for N in N_values}


errors = {}
for N in N_values:
    errors[N] = {
        "Error Absoluto": np.abs(y_sigmoid - y_maclaurin[N]),
        "Error Relativo": np.abs((y_sigmoid - y_maclaurin[N]) / y_sigmoid)
    }


plt.figure(figsize=(10, 6))
plt.plot(x_values, y_sigmoid, label="Sigmoide Exacta", linewidth=2)
for N in N_values:
    plt.plot(x_values, y_maclaurin[N], linestyle='--', label=f"Maclaurin (N={N})")
plt.xlabel("x")
plt.ylabel("Salida del Perceptrón")
plt.title("Salida del Perceptrón con Aproximación de Maclaurin")
plt.legend()
plt.grid()
plt.show()

error_df = pd.DataFrame({"x": x_values, "Sigmoide Exacta": y_sigmoid})
for N in N_values:
    error_df[f"Maclaurin N={N}"] = y_maclaurin[N]
    error_df[f"Error Absoluto N={N}"] = errors[N]["Error Absoluto"]
    error_df[f"Error Relativo N={N}"] = errors[N]["Error Relativo"]


print(error_df.head())