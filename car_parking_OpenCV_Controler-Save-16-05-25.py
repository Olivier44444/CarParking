import cv2
import numpy as np
import math
import time
import heapq

"""
                A faire

> Vérifier comment fonctionnent prepare_obstacle_distance_map() et check_circular_collision_distance_2_circles()
    > Comprendre ce qu'est dist_map, dist_transform
        h, w = dist_map.shape # pas besoin de mettre le [:2]
> Améliorer la fonction de coût du A*, il faut prendre en compte theta
> Regarder si on ne peut pas définir qu'une seule fois les centre_x, radius ...


> Bien regarder comment fonctionne le Hybrid A* (notamment les continue)
"""


# Choix du mode : 
# True : Mode de contrôle de la voiture avec les touches du clavier
# False : Mode car parking avec graphs
Control_Mode = True

# Paramètres de la voiture
Car_length = 20  # en pixels
Car_width = 10   # en pixels
L_essieux = 18   # distance entre les essieux (pixels)

# Exemple 1
#start = 150, 200, 0
#goal = 250, 150, np.pi/2

# Exemple 2
#start = 150, 300, np.pi/2
#goal = 250, 200, 0

# Exemple 3
start = 180, 140, -np.pi/2
goal = 250, 150, np.pi/2

# Exemple 4 : Créneau
start = 225, 170, 0
goal = 180, 150, 0

# Position initiale de la voiture
x = 180  
y = 150  
theta_deg = 0  # angle en degrés (0 = vers la droite)
theta = theta_deg*2*np.pi/360 # angle en radian

radius = int(math.hypot(Car_length, Car_width) / 2) + 2

# Compute circles

center_x = int(x + (Car_length / 4) * math.cos(theta))
center_y = int(y - (Car_width / 4) * math.sin(theta))

# Petits cercles
radius_small = radius // 2 

center_x_1 = int(center_x - radius_small * math.cos(theta))
center_y_1 = int(center_y + radius_small * math.sin(theta))

center_x_2 = int(center_x + radius_small * math.cos(theta))
center_y_2 = int(center_y - radius_small * math.sin(theta))

# Chargement de la map 
map_img = cv2.imread('mapLittle.png')
map_height, map_width, map_channels = map_img.shape
map_shape = [map_height, map_width]



########################################## Fonctions ##########################################



def draw_car(img, x, y, theta):
    """
    Dessine la voiture comme un rectangle orienté autour de l’essieu arrière.
    • img   : 
    • x     : abscisse de la voiture
    • y     : ordonnée de la voiture
    • theta : orientation de la voiture (theta = 0 → voiture vers la droite)
    """

    # Calcul du centre de la voiture par rapport à l’essieu arrière, le rectangle étant dessiné à partir du centre géométrique de la voiture
    center_x = x + (Car_length/4) * math.cos(theta)
    center_y = y - (Car_width/4) * math.sin(theta)

    rect = ((center_x, center_y), (Car_length, Car_width), -np.degrees(theta))
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    car_img = img.copy()
    cv2.drawContours(car_img, [box], 0, (0, 0, 255), -1)  # rouge

    # Afficher l’essieu arrière (centre de rotation) en vert
    cv2.circle(car_img, (int(x), int(y)), 2, (0, 255, 0), -1)

    return car_img



def move_car(x, y, theta, v, delta, dt=1.0):
    """
    Met à jour la position de la voiture en utilisant les équations basées sur le modèle de mouvement
    Input:
    • x     : abscisse de la voiture
    • y     : ordonnée de la voiture
    • theta : orientation de la voiture (theta = 0 → voiture vers la droite)
    • v     : vitesse de la voiture
    • delta : angle de braquage de la voiture
    • dt    : petit pas de temps

    Output:
    Nouvelles positions : 
    • x
    • y
    • theta
    """
    x += v * math.cos(theta) * dt
    y -= v * math.sin(theta) * dt
    theta += (v / L_essieux) * math.tan(delta) * dt

    return x, y, theta



def prepare_obstacle_distance_map(img):
    """Crée une carte de distances aux obstacles à partir d'une image.
    Input:
    • img               : 

    Output:
    
    • dist_transform    : 
    """

    # Détection des obstacles (noir ou vert)
    is_black = np.all(img < [50, 50, 50], axis=-1)
    is_green = np.all(np.logical_and(img >= [0, 100, 0], img <= [100, 255, 100]), axis=-1)

    obstacle_mask = np.logical_or(is_black, is_green).astype(np.uint8) * 255

    # Assurer que l'image est mono-canal
    if len(obstacle_mask.shape) == 3:
        obstacle_mask = cv2.cvtColor(obstacle_mask, cv2.COLOR_BGR2GRAY)

    # Inverser les obstacles : 0 pour obstacles, 255 pour espace libre
    free_space = cv2.bitwise_not(obstacle_mask)

    # Calcul de la distance aux obstacles
    dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)

    return dist_transform


def compute_circles():
    # Grand cercle
    radius = int(math.hypot(Car_length, Car_width) / 2) + 2

    center_x = int(x + (Car_length / 4) * math.cos(theta))
    center_y = int(y - (Car_width / 4) * math.sin(theta))

    # Petits cercles
    radius_small = radius // 2 

    center_x_1 = int(center_x - radius_small * math.cos(theta))
    center_y_1 = int(center_y + radius_small * math.sin(theta))

    center_x_2 = int(center_x + radius_small * math.cos(theta))
    center_y_2 = int(center_y - radius_small * math.sin(theta))
    
    return center_x, center_y, center_x_1, center_y_1, center_x_2, center_y_2

"""
Besoin de compute_circles() dans : 
- check_circular_collision_distance_2_circles()
- La boucle de contrôle

"""



def check_circular_collision_distance_2_circles(dist_map, x, y, theta):
    """
    Détecte une collision basée sur la distance à l'obstacle à trois points (grand cercle + deux petits cercles).
    """
    #print(dist_map.shape[:2])
    #exit()

    Suspected_collision = False
    Collision = False

    # Définition des rayons des cercles
    radius = int(math.hypot(Car_length, Car_width) / 2) + 2 # Grand cercle
    radius_small = radius // 2 # Petits cercles

    # Définition des centres des cercles

    # Grand cercle
    center_x = int(x + (Car_length / 4) * math.cos(theta))
    center_y = int(y - (Car_width / 4) * math.sin(theta))

    # Petit cerlce 1
    center_x_1 = int(center_x - radius_small * math.cos(theta))
    center_y_1 = int(center_y + radius_small * math.sin(theta))

    # Petit cerlce 2
    center_x_2 = int(center_x + radius_small * math.cos(theta))
    center_y_2 = int(center_y - radius_small * math.sin(theta))

    h, w = dist_map.shape # pas besoin de mettre le [:2]

    def distance_at(cx, cy):
        if 0 <= cx < w and 0 <= cy < h:
            value = dist_map[cy, cx]
            if isinstance(value, np.ndarray):
                return value[0]
            return value
        return float('inf')  # Hors image = pas de collision

    # Vérification de collision avec le grand cercle
    if distance_at(center_x, center_y) < radius:
        Suspected_collision = True

        # Vérification de collision avec les petits cercles
        if distance_at(center_x_1, center_y_1) < radius_small or distance_at(center_x_2, center_y_2) < radius_small:
            Collision = True


    # Créaction des cercles

    # On génère le grand cercle en bleu
    cv2.circle(display_img, (center_x, center_y), radius, (255, 0, 0), 2)  # Bleu
    
    # Vérifier collision suspectée
    
    if Collision == True:
        print("❌ Collision détectée")

        # Si une collision est détectée, le grand cercle passe en rouge et le petits cercles aussi
        cv2.circle(display_img, (center_x, center_y), radius, (0, 0, 255), 2)            # Rouge
        cv2.circle(display_img, (center_x_1, center_y_1), radius_small, (0, 0, 255), 2)  # Rouge
        cv2.circle(display_img, (center_x_2, center_y_2), radius_small, (0, 0, 255), 2)  # Rouge
    
    elif Suspected_collision == True:
        print("⚠️ Collision suspectée")
        # Si une collision est suspectée, c'est simplement le grand cercle qui passe en rouge, les petits cercles sont en orange 

        cv2.circle(display_img, (center_x, center_y), radius, (0, 0, 255), 2)              # Rouge
        cv2.circle(display_img, (center_x_1, center_y_1), radius_small, (0, 165, 255), 2)  # Orange
        cv2.circle(display_img, (center_x_2, center_y_2), radius_small, (0, 165, 255), 2)  # Orange
    

    return Suspected_collision, Collision




########################################## Contrôle de la voiture ##########################################


# Générer la carte de distance
dist_map = prepare_obstacle_distance_map(map_img)

display_img = draw_car(map_img, x, y, theta)



if Control_Mode == True:
    # --- Boucle principale ---
    
    while True:
        display_img = draw_car(map_img, x, y, theta)
        

        Suspected_collision, Collision = check_circular_collision_distance_2_circles(dist_map, x, y, theta)
        """
        # Afficher le cercle bleu au centre de la voiture (grande zone de détection)
        center_x = int(x + (Car_length / 4) * math.cos(theta))
        center_y = int(y - (Car_length / 4) * math.sin(theta))
        radius = int(math.hypot(Car_length / 2, Car_width / 2)) + 2

        # Calcul des petits cercles (orange)
        radius_small = radius // 2
        center_x_1 = int(center_x - radius_small* math.cos(theta))
        center_x_2 = int(center_x + radius_small* math.cos(theta))
        center_y_1 = int(center_y + radius_small* math.sin(theta))
        center_y_2 = int(center_y - radius_small* math.sin(theta))
        
        # On génère le grand cercle en bleu
        cv2.circle(display_img, (center_x, center_y), radius, (255, 0, 0), 2)  # Bleu
        
        # Vérifier collision suspectée
        
        if Collision == True:
            print("❌ Collision détectée")

            # Si une collision est détectée, le grand cercle passe en rouge et le petits cercles aussi
            cv2.circle(display_img, (center_x, center_y), radius, (0, 0, 255), 2)            # Rouge
            cv2.circle(display_img, (center_x_1, center_y_1), radius_small, (0, 0, 255), 2)  # Rouge
            cv2.circle(display_img, (center_x_2, center_y_2), radius_small, (0, 0, 255), 2)  # Rouge
        
        elif Suspected_collision == True:
            print("⚠️ Collision suspectée")
            # Si une collision est suspectée, c'est simplement le grand cercle qui passe en rouge, les petits cercles sont en orange 

            cv2.circle(display_img, (center_x, center_y), radius, (0, 0, 255), 2)              # Rouge
            cv2.circle(display_img, (center_x_1, center_y_1), radius_small, (0, 165, 255), 2)  # Orange
            cv2.circle(display_img, (center_x_2, center_y_2), radius_small, (0, 165, 255), 2)  # Orange
        
        """

        # Affichage

        cv2.imshow('Car Parking Simulation', display_img)

        # Lecture de touche
        key = cv2.waitKey(100)
        if key == 27:  # ÉCHAP
            break

        # Commandes de mouvement
        move_map = {
            ord('z'): (5, 0),
            ord('s'): (-5, 0),
            ord('a'): (5, np.radians(30)),
            ord('e'): (5, np.radians(-30)),
            ord('q'): (-5, np.radians(30)),
            ord('d'): (-5, np.radians(-30)),
        }
        if key in move_map:
            v, delta = move_map[key]
            x, y, theta = move_car(x, y, theta, v, delta)

    cv2.destroyAllWindows()



########################################## Implémentation graphs ##########################################


import math
import heapq
from dubins_path_planner import dubins_path_planner


L_essieux = 18.0
rayon_min = 10.0
reverse_penalty = 5.0
turning_penalty = 2.0

class Node3D:
    def __init__(self, x, y, theta, g=0.0, h=0.0, parent=None, direction=1):
        self.x = x
        self.y = y
        self.theta = theta
        self.g = g
        self.h = h
        self.parent = parent
        self.direction = direction  # +1 marche avant, -1 marche arrière

    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f() < other.f()

    def to_key(self):
        return (int(self.x), int(self.y), int(self.theta * 180 / math.pi) % 360)

def heuristic_dubins(current, goal):
    qs = [current.x, current.y, current.theta]
    qe = [goal.x, goal.y, goal.theta]
    turning_radius = rayon_min
    try:
        _, _, _, _, clen = dubins_path_planner(qs, qe, turning_radius)
        return sum(clen)  # longueur totale de l’arc Dubins
    except:
        return float('inf')  # au cas où le chemin serait invalide

def cost_motion(parent, child):
    dist = math.hypot(child.x - parent.x, child.y - parent.y)
    angle_diff = abs((child.theta - parent.theta + math.pi) % (2 * math.pi) - math.pi)
    cost = dist
    if child.direction == -1:
        cost += reverse_penalty
    if angle_diff > 0.01:
        cost += turning_penalty * angle_diff
    return parent.g + cost


######################################## A modifier
# On peut réutiliser les fonctions plus haut

def create_successors(node):
    successors = []
    for direction in [1, -1]:
        for delta_deg in [-30, 0, 30]:
            delta = math.radians(delta_deg)
            v = direction * 5.0
            x = node.x + v * math.cos(node.theta)
            y = node.y - v * math.sin(node.theta)
            new_theta = node.theta + (v / L_essieux) * math.tan(delta)
            successors.append(Node3D(x, y, new_theta, parent=node, direction=direction))
    return successors

def is_goal_reached(n, goal, pos_thresh=5.0, angle_thresh=0.2):
    dx, dy = n.x - goal.x, n.y - goal.y
    dtheta = abs((n.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi)
    return math.hypot(dx, dy) < pos_thresh and dtheta < angle_thresh

def reconstruct_path(node):
    path = []
    while node:
        path.append((node.x, node.y, node.theta))
        node = node.parent
    return path[::-1]

def hybrid_a_star(start, goal, dist_map, map_shape):
    open_list = []
    closed_set = set()

    start_node = Node3D(*start)
    goal_node = Node3D(*goal)
    start_node.h = heuristic_dubins(start_node, goal_node)

    heapq.heappush(open_list, start_node)

    while open_list:
        current = heapq.heappop(open_list)
        key = current.to_key()

        if key in closed_set:
            continue
        closed_set.add(key)

        if is_goal_reached(current, goal_node):
            return reconstruct_path(current)

        for neighbor in create_successors(current):
            ix, iy = int(neighbor.x), int(neighbor.y)
            if not (0 <= ix < map_shape[1] and 0 <= iy < map_shape[0]):
                continue
            if dist_map[iy, ix] < 2:
                continue

            neighbor.g = cost_motion(current, neighbor)
            neighbor.h = heuristic_dubins(neighbor, goal_node)
            heapq.heappush(open_list, neighbor)

    return None





path = hybrid_a_star(start=start,
                     goal=goal,
                     dist_map=dist_map,
                     map_shape=map_shape)

#print(path)



########################################## Affichage du path ##########################################





def replay_path_on_map(map_img, path, delay=100):
    """
    Affiche la voiture qui suit un chemin donné (path).
    - map_img : image de fond
    - path : liste de tuples (x, y, theta)
    - delay : délai entre les frames en millisecondes (default: 100 ms)
    """
    for (x, y, theta) in path:
        display_img = draw_car(map_img, x, y, theta)

        # --- Cercle principal ---
        center_x = int(x + (Car_length / 4) * math.cos(theta))
        center_y = int(y - (Car_length / 4) * math.sin(theta))
        radius = int(math.hypot(Car_length / 2, Car_width / 2)) + 2
        cv2.circle(display_img, (center_x, center_y), radius, (255, 0, 0), 2)

        # --- Petits cercles ---
        radius_small = radius // 2
        center_x_1 = int(center_x - radius_small * math.cos(theta))
        center_x_2 = int(center_x + radius_small * math.cos(theta))
        center_y_1 = int(center_y + radius_small * math.sin(theta))
        center_y_2 = int(center_y - radius_small * math.sin(theta))

        cv2.circle(display_img, (center_x_1, center_y_1), radius_small, (0, 165, 255), 1)
        cv2.circle(display_img, (center_x_2, center_y_2), radius_small, (0, 165, 255), 1)

        # Affichage
        cv2.imshow('Replay Car Path', display_img)
        key = cv2.waitKey(delay)
        if key == 27:  # ÉCHAP pour arrêter
            break

    # 🔒 Garde la dernière image affichée jusqu’à appui sur une touche
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if Control_Mode == False:
    replay_path_on_map(map_img, path, delay=100)
