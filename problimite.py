import numpy as np
from tridiagonal import tridiagonal 

def problimite(N, Q, R, a, b, alpha, beta):
    h = (b - a) / (N + 1)
    
    # Q et R doivent être des tableaux NumPy pour que l'opération vectorisée fonctionne
    #Q = np.array(Q, dtype=float)
    #R = np.array(R, dtype=float)
    
    D = -2.0 - Q * (h**2)
    I = np.ones(N - 1)
    S = np.ones(N - 1)

    B = R * (h**2)
    B[0] = B[0] - alpha
    B[-1] = B[-1] - beta 

    # Appel au solveur
    y1 = tridiagonal(N, D, I, S, B)
    
    # Concaténation de alpha, du vecteur solution y1, et beta
    x = np.concatenate(([alpha], y1, [beta]))

    return x