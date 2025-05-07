import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paramètres du modèle
L = 2.5  # Empattement (mètres)
dt = 0.1  # Pas de temps (secondes)
T = 20    # Durée totale (secondes)

# Vitesse constante
v = 5.0  # m/s

# Conditions initiales
x, y, theta = 0.0, 0.0, 0.0

# Listes pour stocker la trajectoire
x_list = [x]
y_list = [y]

# Fonction pour définir l'angle de braquage selon le temps
def steering_angle(t):
    if t < 5:
        return np.deg2rad(0)  # Tout droit pendant 5 secondes
    elif t < 10:
        return np.deg2rad(20)  # Tourner à gauche pendant 5 secondes
    else:
        return np.deg2rad(-20)  # Tourner à droite pendant 10 secondes

# Préparation de la figure
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'b-', lw=2)
car, = ax.plot([], [], 'ro')  # un point rouge pour représenter la voiture

ax.set_xlim(-10, 100)
ax.set_ylim(-30, 30)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Trajectoire de la voiture animée')
ax.grid()
ax.set_aspect('equal')

# Fonction d'initialisation pour l'animation
def init():
    line.set_data([], [])
    car.set_data([], [])
    return line, car

# Fonction d'animation appelée à chaque frame
def animate(i):
    global x, y, theta
    t = i * dt

    delta = steering_angle(t)

    # Mise à jour de la position et orientation
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(delta) * dt

    x_list.append(x)
    y_list.append(y)

    line.set_data(x_list, y_list)
    car.set_data(x, y)

    return line, car

# Création de l'animation
ani = animation.FuncAnimation(fig, animate, frames=int(T/dt), init_func=init,
                              interval=100, blit=True)

plt.show()
