import cv2
import numpy as np
import heapq
from dubins_path_planner import dubins_path_planner
import matplotlib.pyplot as plt


"""
#################################################################   To run the script   #################################################################

Activate the environment car_env :
>   source car_env/bin/activate

Installer les librairies : 
>   pip install opencv-python numpy

"""


""" Choix du mode : """ 
# True : Mode de contrôle de la voiture avec les touches du clavier
# False : Mode car parking avec graphs
Control = False
Show_Circle = True # Pour afficher les cercles de détection d'obstacles
Show_Trace = True # Afficher les positions précédentes de la voiture



########################################## Paramètres ##########################################

Obstacles = [[0,0,0], [0,127,0]] # liste de pixels obstacles

Car_length = 20  # (pixels)
Car_width = 10   # (pixels)
L_essieux = 18   # distance entre les essieux (pixels)

phi_deg = 30 # angle de braquage (degrés)
phi = np.radians(phi_deg) # (radians)
abs_v = 5.0 # vitesse de la voiture en valeur absolue
dt = 1.0 # pas de temps

angular_precision_deg = 5 # (deg)
angular_precision = np.radians(angular_precision_deg) # (radians)
position_precision = 1.0 # (pixels)


rayon_min = L_essieux / np.tan(phi)/2 # Rayon minimal pour calculer le chemin de Dubins

# Pénalités pour calculer le coût
reverse_penalty = 5.0
turning_penalty = 2.0
turning_penalty_condition = 0.01

delay = 100 # frames (Graph Mode)

main_color = (220, 0, 0) # Couleur de la voiture

""" Exemples de points de départ et d'arrivé (Graph Mode) 
x_s, y_s, theta_s : coordonnées du start
x_g, y_g, theta_g : coordonnées du goal

Attention : theta_s et theta_g en degrés
"""


# Exemple 1 : Trajets classiques

#x_s, y_s, theta_s = 150, 200, 0
#x_g, y_g, theta_g = 250, 150, 90

# Exemple 2
#x_s, y_s, theta_s = 150, 300, 90
#x_g, y_g, theta_g = 250, 200, 0

# Exemple 3 : Demi tour
#x_s, y_s, theta_s = 180, 140, -90
#x_g, y_g, theta_g = 250, 150, 90

# Exemple 4 : Créneau
x_s, y_s, theta_s = 225, 170, 0
x_g, y_g, theta_g = 180, 150, 0

# Exemple 5 : Long trajet
#x_s, y_s, theta_s = 300, 210, 0
#x_g, y_g, theta_g = 450, 250, -90

start = x_s, y_s, np.radians(theta_s)
goal = x_g, y_g, np.radians(theta_g)

""" Position initiale de la voiture (Contrôle mode) """
x, y, theta_deg = 300, 210, 0 # angle en degrés (0 = vers la droite)

theta = np.radians(theta_deg) # (radians)


# Chargement de la map 
map_img = cv2.imread('mapLittle.png')
map_height, map_width, map_channels = map_img.shape
map_shape = [map_height, map_width]





########################################## Preprocessing de l'image ##########################################


def show_colors():
    img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)

    pixels = img_rgb.reshape(-1, 3) 
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    fig_height = max(5, len(unique_colors) * 0.6)
    fig, (ax_img, ax_legend) = plt.subplots(1, 2, figsize=(10, fig_height))

    ax_img.imshow(img_rgb)
    ax_img.set_title("Carte (image d'origine)")
    ax_img.axis('off')
    
    square_size = 1
    for i, color in enumerate(unique_colors):
        rgb = tuple(int(c) for c in color)
        ax_legend.add_patch(
            plt.Rectangle((0, i), square_size, square_size, color=np.array(rgb)/255.0)
        )
        ax_legend.text(square_size + 0.2, i + 0.5, str(rgb), va='center', fontsize=10)

    ax_legend.set_xlim(0, 4)
    ax_legend.set_ylim(0, len(unique_colors))
    ax_legend.invert_yaxis() 
    ax_legend.axis('off')
    ax_legend.set_title("Légende des couleurs")

    plt.tight_layout()
    plt.show()

#show_colors()




########################################## Fonctions Principales ##########################################


def draw_car(img, x, y, theta, color=main_color):
    """
    Dessine la voiture comme un rectangle orienté autour de l’essieu arrière.

    Input :
    • img : image de la map
    • x : abscisse de la voiture
    • y : ordonnée de la voiture
    • theta : orientation de la voiture (theta = 0 → voiture vers la droite)
    • color : couleur de la voiture
    Output : 
    • car_img : image avec la voiture
    """

    # Calcul du centre de la voiture par rapport à l’essieu arrière, le rectangle étant dessiné à partir du centre géométrique de la voiture
    center_x = x + (Car_length/4) * np.cos(theta)
    center_y = y - (Car_width/4) * np.sin(theta)

    rect = ((center_x, center_y), (Car_length, Car_width), -np.degrees(theta))
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    car_img = img.copy()
    cv2.drawContours(car_img, [box], 0, color, -1) 

    # Afficher l’essieu arrière (centre de rotation) en vert
    cv2.circle(car_img, (int(x), int(y)), 1, (0, 255, 0), -1)

    return car_img




def move_car(x, y, theta, v, delta, dt=1.0):
    """
    Met à jour la position de la voiture en utilisant les équations basées sur le modèle de mouvement
    Input:
    • x : abscisse de la voiture
    • y : ordonnée de la voiture
    • theta : orientation de la voiture (theta = 0 → voiture vers la droite) (radian)
    • v : vitesse de la voiture
    • delta : angle de braquage de la voiture
    • dt : petit pas de temps
    Output:
    Nouvelles positions : 
    • x
    • y
    • theta
    """
    x += v * np.cos(theta) * dt
    y -= v * np.sin(theta) * dt
    theta += (v / L_essieux) * np.tan(delta) * dt

    return x, y, theta


def generate_obstacle_distance_map(img):
    """Crée une carte de distances aux obstacles à partir d'une image.
    Input:
    • img : map affichée avec OpenCV (BGR)
    Output:
    • dist_transform : carte des distances (array)
    """

    obstacle_mask = np.zeros(img.shape[:2], dtype=bool)

    for color in Obstacles: # Comparaison avec les pixels de la map
        match = np.all(img == color, axis=-1)
        obstacle_mask = np.logical_or(obstacle_mask, match)

    obstacle_mask = obstacle_mask.astype(np.uint8) * 255

    if len(obstacle_mask.shape) == 3: # Monocanal
        obstacle_mask = cv2.cvtColor(obstacle_mask, cv2.COLOR_BGR2GRAY)

    free_space = cv2.bitwise_not(obstacle_mask) # Les obstacles doivent être en noir

    dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5) # distanceTransform permet alors de calculer la distance de chaque pixel au pixel noir leplus proche

    return dist_transform


def generate_obstacle_distance_map(img):
    """Crée une carte de distances aux obstacles à partir d'une image.
    Input:
    • img : map affichée avec OpenCV
    Output:
    • dist_transform : carte des distances (array)
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


def collision_detector(img, dist_map, x, y, theta, circle_condition):
    """Détecte une collision basée sur la distance à l'obstacle à trois points (grand cercle + deux petits cercles).
    Input:
    • img : image avec la voiture (issue de draw_car)
    • dist_map : carte des distances (np.array)
    • x : (pixels)
    • y : (pixels)
    • theta : (rad)
    • circle_condition : (Show_Circle) booléen qui active ou non l'affichage des cercles de détection d'obstacles
    Output:
    • Collision : True / False s'il y a une collision (bool)
    • img_out : image avec la voiture et les cercles de détection de collision
    """


    img_out = img.copy()

    Suspected_collision = False
    Collision = False

    # Définition des rayons des cercles
    radius = int(np.hypot(Car_length, Car_width) / 2) + 2 # Grand cercle
    radius_small = radius // 2 # Petits cercles

    # Définition des centres des cercles

    # Grand cercle
    center_x = int(x + (Car_length / 4) * np.cos(theta))
    center_y = int(y - (Car_width / 4) * np.sin(theta))

    # Petit cerlce 1
    center_x_1 = int(center_x - radius_small * np.cos(theta))
    center_y_1 = int(center_y + radius_small * np.sin(theta))

    # Petit cerlce 2
    center_x_2 = int(center_x + radius_small * np.cos(theta))
    center_y_2 = int(center_y - radius_small * np.sin(theta))

    h, w = dist_map.shape

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
    if circle_condition == True:

        # On génère le grand cercle en bleu
        cv2.circle(img_out, (center_x, center_y), radius, (255, 0, 0), 1)  # Bleu
        
        # Vérifier collision suspectée
        
        if Collision == True:
            #print("Collision détectée")

            # Si une collision est détectée, le grand cercle passe en rouge et le petits cercles aussi
            cv2.circle(img_out, (center_x, center_y), radius, (0, 0, 255), 1)            # Rouge
            cv2.circle(img_out, (center_x_1, center_y_1), radius_small, (0, 0, 255), 1)  # Rouge
            cv2.circle(img_out, (center_x_2, center_y_2), radius_small, (0, 0, 255), 1)  # Rouge
        
        elif Suspected_collision == True:
            #print("Collision suspectée")
            # Si une collision est suspectée, c'est simplement le grand cercle qui passe en rouge, les petits cercles sont en orange 

            cv2.circle(img_out, (center_x, center_y), radius, (0, 0, 255), 1)              # Rouge
            cv2.circle(img_out, (center_x_1, center_y_1), radius_small, (0, 165, 255), 1)  # Orange
            cv2.circle(img_out, (center_x_2, center_y_2), radius_small, (0, 165, 255), 1)  # Orange
        
    
    return Collision, img_out




########################################## Contrôle de la voiture ##########################################



def Control_Mode(x, y, theta): 
    """Permet de contrôler la voiture.
    Input:
    • x
    • y
    • theta
    Output:
    • Affiche la map avec la voiture tant qu'on n'appuie pas sur Echap
    """
    while True:

        # Bloc d'affichage
        display_img = draw_car(map_img, x, y, theta, main_color)
        Collision, img_out = collision_detector(display_img, dist_map, x, y, theta, Show_Circle)
        cv2.imshow('Car Parking Simulation', img_out)

        # Lecture de touche
        key = cv2.waitKey(100)
        if key == 27:  # ÉCHAP
            break

        # Commandes de mouvement
        move_map = {
            ord('z'): (abs_v, 0),
            ord('s'): (-abs_v, 0),
            ord('a'): (abs_v, phi),
            ord('e'): (abs_v, -phi),
            ord('q'): (-abs_v, phi),
            ord('d'): (-abs_v, -phi),
        }
        if key in move_map:
            v, delta = move_map[key]
            x, y, theta = move_car(x, y, theta, v, delta)

    cv2.destroyAllWindows()



########################################## Implémentation graphs ##########################################

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

    def __lt__(self, other): # less than : Utilisé avec heapq comme opérateur "<"
        return self.f() < other.f()

    def to_key(self):
        return (int(self.x), int(self.y), int(self.theta * 180 / np.pi) % 360)

    

def heuristic_dubins(current, goal, turning_radius):
    """Calcule l'heuristic h.
    Input:
    • current : noeud courant
    • goal : noeud 
    • turning_radius : rayon utilisé dans le calcul des chemins de Dubins
    Output:
    • h : longueur du chemin de Dubins 
    """

    qs = [current.x, current.y, current.theta]
    qe = [goal.x, goal.y, goal.theta]
    
    result = dubins_path_planner(qs, qe, turning_radius)

    if result is None:
        return float('inf')  # Aucun chemin valide trouvé

    px, py, pyaw, path_type, arc_len = result
    return sum(arc_len)  # Longueur totale du chemin Dubins



def cost_motion(parent, child):
    """Calcule le coût g.
    Input:
    • parent : noeud parent
    • child : noeud enfant
    Output:
    • g : coût du noeud de départ jusqu'au noeud courant
    """

    dist = abs_v * dt
    angle_diff = abs((child.theta - parent.theta + np.pi) % (2 * np.pi) - np.pi)
    cost = dist

    # Ajout des pénalités
    if child.direction == -1:
        cost += reverse_penalty
    if angle_diff > turning_penalty_condition:
        cost += turning_penalty * angle_diff

    return parent.g + cost


def create_neighbor(node):
    """Génère les voisins d'un noeud.
    Input:
    • node : noeud quelconque du graph
    Output:
    • sum(arc_len) : longueur du chemin de Dubins 
    """
    successors = []
    for direction in [1, -1]:
        for delta in [-phi, 0, phi]:
            v = direction * abs_v
            x, y, theta = move_car(node.x, node.y, node.theta, v, delta, dt=dt)
            Collision, img_out = collision_detector(display_img, dist_map, x, y, theta, Show_Circle)
            if (0 <= x < map_shape[1] and 0 <= y < map_shape[0]) and Collision == False:
                successors.append(Node3D(x, y, theta, parent=node, direction=direction))
    return successors


def is_goal_reached(current, goal, pos_prec=position_precision, ang_prec=angular_precision):
    """Vérifie si la cible a été atteinte.
    Input:
    • current : noeud courant
    • goal : noeud cible
    • pos_prec : précision suivant x et y (plus ce coefficient est faible, plus on souhaite se rapprocher de la cible)
    • ang_prec : précision angulaire (idem)
    Output:
    • (Bool): True si le goal est atteint
    """

    dx, dy = current.x - goal.x, current.y - goal.y
    dtheta = abs((current.theta - goal.theta + np.pi) % (2 * np.pi) - np.pi)
    return np.hypot(dx, dy) < pos_prec and dtheta < ang_prec


def reconstruct_path(node):
    """Reconstruit le chemin pour arriver à node.
    Input:
    • node : noeud
    Output:
    • path[::-1] : liste des noeuds menants à node 
    """

    path = []
    while node:
        path.append((node.x, node.y, node.theta))
        node = node.parent
    return path[::-1]


def hybrid_a_star(start, goal):
    """Calcule le chamin le plus du noeud de départ jusqu'au noeud d'arrivée.
    Input:
    • start : noeud de départ
    • goal : noeud cible
    Output:
    • path : chemin le plus court entre start et goal
    """

    open_list = []
    closed_set = set()

    start_node = Node3D(*start)
    goal_node = Node3D(*goal)
    start_node.h = heuristic_dubins(start_node, goal_node, rayon_min)

    heapq.heappush(open_list, start_node)

    while open_list:
        current = heapq.heappop(open_list)
        key = current.to_key()

        if key in closed_set:
            continue
        closed_set.add(key)

        if is_goal_reached(current, goal_node):
            return reconstruct_path(current)

        for neighbor in create_neighbor(current):
            neighbor.g = cost_motion(current, neighbor)
            neighbor.h = heuristic_dubins(neighbor, goal_node, rayon_min)
            heapq.heappush(open_list, neighbor)

    return None


########################################## Affichage du path ##########################################


def replay_path_on_map(map_img, delay, start, goal, dist_map):
    """Affiche le déplacement de la voiture suivant le chemin le plus court sur la carte. 
    Input:
    • map_img : map ouverte avec OpenCV
    • delay : temps de latence entre les frames
    • start : noeud de départ
    • goal : noeud cible
    • dist_map : carte des distances
    Output:
    • Affiche la map
    """
        
    path = hybrid_a_star(start, goal)

    trace_img = map_img.copy()

    for i in range(len(path)):
        x, y, theta = path[i]

        if Show_Trace == True:

            # Calcule une couleur en fonction de la progression
            alpha = i / len(path)
            blue_intensity = int(220 * (1 - alpha))          # diminue au fil du temps
            color = (220, blue_intensity, blue_intensity)    # dégradé vers le bleu clair

            trace_img = draw_car(trace_img, x, y, theta, color=color)

        display_img = draw_car(trace_img, x, y, theta, main_color)  # voiture actuelle 
        Collision, img_out = collision_detector(display_img, dist_map, x, y, theta, Show_Circle)

        cv2.imshow('Trajectoire de la voiture', img_out)
        if cv2.waitKey(delay) == 27:
            break

    

    # Garde la dernière image affichée jusqu’à appui sur une touche
    cv2.waitKey(0)
    cv2.destroyAllWindows()


########################################## Preprocessing ##########################################

# Générer la carte de distance
dist_map = generate_obstacle_distance_map(map_img)
display_img = draw_car(map_img, x, y, theta, main_color)

print(type(dist_map))


########################################## Affichage ##########################################


if Control == True:
    Control_Mode(x, y, theta)

if Control == False:
    replay_path_on_map(map_img, delay, start, goal, dist_map)
