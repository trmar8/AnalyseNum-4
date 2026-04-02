import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite

# ==========================================
# Paramètres du problème (Question 2)
# ==========================================
L = 6.0          
Ta = 20.0        
T_init = 350.0   
k = 1.2          

a = 0.0          
b = L            
alpha = T_init   
beta = Ta        

# L'équation est y''(x) - k^2 y(x) = -k^2 Ta
# Donc q(x) = k^2 et r(x) = -k^2 Ta
q_val = k**2
r_val = -(k**2) * Ta

def solution_exacte(x):
    """Calcule la solution analytique exacte."""
    B2 = (T_init - Ta) / (1 - np.exp(-2 * k * L))
    B1 = (T_init - Ta) - B2
    return Ta + B1 * np.exp(k * x) + B2 * np.exp(-k * x)

# ==========================================
# Question 2a) : Graphique des approximations
# ==========================================

# Cas h = 2
h1 = 2.0
N1 = int(round((b - a) / h1)) - 1
Q1 = np.full(N1, q_val)
R1 = np.full(N1, r_val)
y_approx_h1 = problimite(N1, Q1, R1, a, b, alpha, beta)
x_nodes_h1 = np.linspace(a, b, N1 + 2)

# Cas h = 1
h2 = 1.0
N2 = int(round((b - a) / h2)) - 1
Q2 = np.full(N2, q_val)
R2 = np.full(N2, r_val)
y_approx_h2 = problimite(N2, Q2, R2, a, b, alpha, beta)
x_nodes_h2 = np.linspace(a, b, N2 + 2)

# Grille fine pour la solution exacte
x_exact = np.linspace(a, b, 200)
y_exact = solution_exacte(x_exact)

# Tracé de la Figure 1
plt.figure(1, figsize=(10, 6))
plt.plot(x_exact, y_exact, '-', color='black', label='Solution exacte')
plt.plot(x_nodes_h1, y_approx_h1, 'o--', color='red', label='Approximation (h = 2)')
plt.plot(x_nodes_h2, y_approx_h2, 's--', color='blue', label='Approximation (h = 1)')

plt.xlabel('Position $x$')
plt.ylabel('Température $y(x)$')
plt.title("Figure 1 : Distribution de la température")
plt.legend()
plt.grid(True)

# ==========================================
# Question 2b) : Analyse de l'erreur E(h)
# ==========================================

valeurs_h = [L/3, L/6, L/100, L/10**3, L/10**4]
erreurs = []

for h in valeurs_h:
    N = int(round((b - a) / h)) - 1
    
    Q = np.full(N, q_val)
    R = np.full(N, r_val)
    
    y_approx = problimite(N, Q, R, a, b, alpha, beta)
    
    x_nodes = np.linspace(a, b, N + 2)
    y_exact_nodes = solution_exacte(x_nodes)
    
    erreur_max = np.max(np.abs(y_approx - y_exact_nodes))
    erreurs.append(erreur_max)

# Tracé de la Figure 2 (échelle logarithmique)
plt.figure(2, figsize=(10, 6))
plt.loglog(valeurs_h, erreurs, marker='o', linestyle='-', color='purple', label='Erreur maximale $E(h)$')

# Ajout d'une droite de référence de pente 2 (O(h^2)) pour comparer visuellement
ref_y = (erreurs[0] / (valeurs_h[0]**2)) * np.array(valeurs_h)**2
plt.loglog(valeurs_h, ref_y, '--', color='gray', label='Pente de référence (Ordre 2)')

plt.xlabel('Pas de discrétisation $h$')
plt.ylabel('Erreur maximale $E(h)$')
plt.title("Figure 2 : Erreur d'approximation en fonction de $h$")
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()