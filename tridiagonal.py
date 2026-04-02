import numpy as np

def tridiagonal(N, D, I, S, b):
    # Création de copies pour ne pas modifier les vecteurs originaux
    D = np.array(D, dtype=float).copy()
    I = np.array(I, dtype=float)
    S = np.array(S, dtype=float).copy()
    b = np.array(b, dtype=float)

    y = np.zeros(N)
    x = np.zeros(N)

    # Initialisation (indice 0)
    y[0] = b[0] / D[0]

    # Boucle avant
    for i in range(1, N):
        S[i-1] = S[i-1] / D[i-1]
        D[i] = D[i] - I[i-1] * S[i-1]
        y[i] = (b[i] - I[i-1] * y[i-1]) / D[i]

    # Dernière valeur
    x[N-1] = y[N-1]

    # Boucle arrière
    for i in range(N-2, -1, -1):
        x[i] = y[i] - S[i] * x[i+1]

    return x