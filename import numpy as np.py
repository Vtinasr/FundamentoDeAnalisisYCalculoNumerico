import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Generar 100 valores en un rango adecuado para evaluar la función sigmoide
x_values = np.linspace(-5, 5, 100)

# Definir la función sigmoide exacta
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Evaluar la función sigmoide exacta
y_sigmoid = sigmoid(x_values)

# Aproximación de la sigmoide usando la serie de Maclaurin
def maclaurin_sigmoid(x, N):
    """
    Aproxima la función sigmoide usando la expansión de Maclaurin de exp(-x).
    """
    maclaurin_exp = sum(((-x)**n / math.factorial(n)) for n in range(N))
    return 1 / (1 + maclaurin_exp)

# Evaluar la sigmoide con truncamiento en diferentes niveles de Maclaurin
N_values = [3, 6, 9]  # Diferentes niveles de truncamiento

y_maclaurin = {N: np.array([maclaurin_sigmoid(x, N) for x in x_values]) for N in N_values}

# Cálculo de errores de truncamiento y error relativo
errors = {}
for N in N_values:
    errors[N] = {
        "Error Absoluto": np.abs(y_sigmoid - y_maclaurin[N]),
        "Error Relativo": np.abs((y_sigmoid - y_maclaurin[N]) / y_sigmoid)
    }

# Graficar la función sigmoide exacta y sus aproximaciones
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

# Crear DataFrame con los resultados de errores
error_df = pd.DataFrame({"x": x_values, "Sigmoide Exacta": y_sigmoid})
for N in N_values:
    error_df[f"Maclaurin N={N}"] = y_maclaurin[N]
    error_df[f"Error Absoluto N={N}"] = errors[N]["Error Absoluto"]
    error_df[f"Error Relativo N={N}"] = errors[N]["Error Relativo"]

# Mostrar las primeras filas del DataFrame con los resultados
print(error_df.head())
