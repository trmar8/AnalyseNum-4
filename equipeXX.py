import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite

L = 6.0          
Ta = 20.0        
T = 350.0   
k = 1.2          

a = 0.0          
b = L            
alpha = T   
beta = Ta        

q = k**2
r = -(k**2) * Ta

def solution(x):
    #quand x = 0, y(x)= T
    #quand x = 1, y(x)= Ta
    #0 = B1e^(kL) + B2e^(-kL)
    B2 = (T - Ta) / (1 - np.exp(-2 * k * L))
    B1 = (T - Ta) - B2


    return Ta + B1 * np.exp(k * x) + B2 * np.exp(-k * x)

h1 = 1.0
N1 = int((b - a) / h1) - 1
Q1 = np.full(N1, q)
R1 = np.full(N1, r)
y_approx_h1 = problimite(N1, Q1, R1, a, b, alpha, beta)
x_noeuds_h1 = np.linspace(a, b, N1 + 2)

h2 = 2.0
N2 = int((b - a) / h2) - 1
Q2 = np.full(N2, q)
R2 = np.full(N2, r)
y_approx_h2 = problimite(N2, Q2, R2, a, b, alpha, beta)
x_noeuds_h2 = np.linspace(a, b, N2 + 2)

x_exact = np.linspace(a, b, 1000)
y_exact = solution(x_exact)

plt.figure(1)
plt.plot(x_exact, y_exact, color='black', label='Solution exacte')
plt.plot(x_noeuds_h1, y_approx_h1, 'o--', color='red', label='Approximation (h = 1)')
plt.plot(x_noeuds_h2, y_approx_h2, 's--', color='green', label='Approximation (h = 2)')
plt.xlabel('Position sur la barre (x)')
plt.ylabel('Température y(x)')
plt.title("Figure 1 : Distribution de la température")
plt.legend()
plt.grid(True)

valeurs_h = [L/3, L/6, L/100, L/1000, L/10000]
erreurs = []

for h in valeurs_h:
    N = int((b - a) / h) - 1
    
    Q = np.full(N, q)
    R = np.full(N, r)
    
    y_approx = problimite(N, Q, R, a, b, alpha, beta)
    
    x_noeuds = np.linspace(a, b, N + 2)
    y_exact_noeuds = solution(x_noeuds)
    
    erreur_max = np.max(np.abs(y_approx - y_exact_noeuds))
    erreurs.append(erreur_max)

plt.figure(2)
plt.loglog(valeurs_h, erreurs, marker='o')
plt.xlabel('Pas (h)')
plt.ylabel('Erreur maximale E(h)')
plt.title("Figure 2 : Erreur d'approximation en fonction de $h$")
plt.grid(True)
plt.show()

