import numpy as np
from tridiagonal import tridiagonal 

def problimite(N, Q, R, a, b, alpha, beta):
    h = (b - a) / (N + 1)
    
    D = -2 - Q * (h**2)
    I = np.ones(N - 1)
    S = np.ones(N - 1)

    B = R * (h**2)
    B[0] = B[0] - alpha
    B[-1] = B[-1] - beta 

    y1 = tridiagonal(N, D, I, S, B)
    
    x = np.concatenate(([alpha], y1, [beta]))

    return x