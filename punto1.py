import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


x_values = np.linspace(-5, 5, 100)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_sigmoid = sigmoid(x_values)


def maclaurin_sigmoid(x, N):
    """
    Aproxima la función sigmoide usando la expansión de Maclaurin de exp(-x).
    """
    maclaurin_exp = sum(((-x)**n / math.factorial(n)) for n in range(N))
    return 1 / (1 + maclaurin_exp)


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
plt.ylabel("f(x)")
plt.title("Comparación de la Función Sigmoide y sus Aproximaciones")
plt.legend()
plt.grid()
plt.show()

error_df = pd.DataFrame({"x": x_values, "Sigmoide Exacta": y_sigmoid})
for N in N_values:
    error_df[f"Maclaurin N={N}"] = y_maclaurin[N]
    error_df[f"Error Absoluto N={N}"] = errors[N]["Error Absoluto"]
    error_df[f"Error Relativo N={N}"] = errors[N]["Error Relativo"]

print(error_df.head())
