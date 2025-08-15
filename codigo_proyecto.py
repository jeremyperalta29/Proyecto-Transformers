import numpy as np

# 1. Datos de entrada simulados (cada fila es un "vector palabra")
# Supongamos 3 palabras, cada una representada con 4 valores (embedding)
X = np.array([
    [1, 0, 1, 0],   # Palabra 1
    [0, 2, 0, 2],   # Palabra 2
    [1, 1, 1, 1]    # Palabra 3
], dtype=float)

print("Entrada X (Embeddings):\n", X)

# 2. Pesos simulados para Q, K y V
W_Q = np.random.rand(4, 4)
W_K = np.random.rand(4, 4)
W_V = np.random.rand(4, 4)

# 3. Calculamos Q, K, V
Q = X @ W_Q  # Multiplicación matricial
K = X @ W_K
V = X @ W_V

print("\nMatriz Q:\n", Q)
print("\nMatriz K:\n", K)
print("\nMatriz V:\n", V)

# 4. Calculamos puntuaciones de atención
scores = Q @ K.T  # Producto Q por transpuesta de K
print("\nPuntuaciones de atención (Q·K^T):\n", scores)

# 5. Normalizamos con softmax para que sean proporciones
def softmax(matrix):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    return exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)

attention_weights = softmax(scores)
print("\nPesos de atención (Softmax):\n", attention_weights)

# 6. Multiplicamos por V para obtener la salida
output = attention_weights @ V
print("\nSalida final (Atención aplicada a V):\n", output)
