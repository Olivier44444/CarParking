import cv2
import numpy as np
import math

# --- Paramètres de la voiture ---
CAR_LENGTH = 20  # en pixels
CAR_WIDTH = 10   # en pixels
WHEELBASE = 18   # distance entre les essieux (pixels)

# --- Position initiale ---
x = 170  # position du centre de l’essieu arrière (x)
y = 150  # position du centre de l’essieu arrière (y)
theta = np.pi/2  # angle en radians (0 = vers la droite)

# --- Charger la map ---
map_img = cv2.imread('mapLittle.png')
map_height, map_width, map_channels = map_img.shape
map_shape = [map_height, map_width]

def draw_car(img, x, y, theta):
    """Dessine la voiture comme un rectangle orienté autour de l’essieu arrière."""
    # Calcul du centre de la voiture par rapport à l’essieu arrière
    center_x = x + (CAR_LENGTH/4) * math.cos(theta)
    center_y = y - (CAR_LENGTH/4) * math.sin(theta)

    rect = ((center_x, center_y), (CAR_LENGTH, CAR_WIDTH), -np.degrees(theta))
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    car_img = img.copy()
    cv2.drawContours(car_img, [box], 0, (0, 0, 255), -1)  # rouge

    # Afficher l’essieu arrière (centre de rotation) en vert
    cv2.circle(car_img, (int(x), int(y)), 2, (0, 255, 0), -1)  # point vert

    return car_img



def move_car(x, y, theta, v, delta, dt=1.0):
    """Met à jour la position de la voiture selon le modèle bicycle."""
    x += v * math.cos(theta) * dt
    y -= v * math.sin(theta) * dt
    theta += (v / WHEELBASE) * math.tan(delta) * dt

    return x, y, theta



def prepare_obstacle_distance_map(img):
    """Crée une carte de distances aux obstacles à partir d'une image."""

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



def check_circular_collision_distance_2_circles(dist_map, x, y, theta):
    """Détecte une collision basée sur la distance à l'obstacle à trois points (grand cercle + deux petits cercles)."""

    Suspected_collision = False
    Collision = False

    # ---- Paramètres ----
    radius = int(math.hypot(CAR_LENGTH, CAR_WIDTH) / 2) + 2
    radius_small = radius // 2
    distance_threshold = radius
    distance_threshold_small = radius_small

    # ---- Points à tester (centre grand cercle + deux petits cercles) ----
    center_x = int(x + (CAR_LENGTH / 4) * math.cos(theta))
    center_y = int(y - (CAR_LENGTH / 4) * math.sin(theta))

    center_x_1 = int(center_x - radius_small * math.cos(theta))
    center_y_1 = int(center_y + radius_small * math.sin(theta))

    center_x_2 = int(center_x + radius_small * math.cos(theta))
    center_y_2 = int(center_y - radius_small * math.sin(theta))

    h, w = dist_map.shape[:2]

    def distance_at(cx, cy):
        if 0 <= cx < w and 0 <= cy < h:
            value = dist_map[cy, cx]
            if isinstance(value, np.ndarray):
                return value[0]
            return value
        return float('inf')  # Hors image = pas de collision

    # ---- Vérification du grand cercle ----
    if distance_at(center_x, center_y) < distance_threshold:
        Suspected_collision = True

        # ---- Vérification des deux petits cercles ----
        if distance_at(center_x_1, center_y_1) < distance_threshold_small or \
           distance_at(center_x_2, center_y_2) < distance_threshold_small:
            Collision = True

    return Suspected_collision, Collision




########################################## Contrôle de la voiture ##########################################


# Prétraitement pour la détection de collision
dist_map = prepare_obstacle_distance_map(map_img)

display_img = draw_car(map_img, x, y, theta)



########################################## Implémentation graphs ##########################################

import heapq
import numpy as np
import math

def heuristic(x, y, goal_x, goal_y):
    return math.hypot(goal_x - x, goal_y - y)

def is_in_bounds(x, y, map_shape):
    return 0 <= int(x) < map_shape[1] and 0 <= int(y) < map_shape[0]

def hybrid_a_star(start, goal, dist_map, map_shape):
    open_set = []
    closed_set = set()

    x0, y0, theta0 = start
    xg, yg, _ = goal

    # État initial
    cost = 0
    parent = None
    heapq.heappush(open_set, (cost + heuristic(x0, y0, xg, yg), cost, (x0, y0, theta0), parent))

    resolutions = {
        "x": 1, "y": 1, "theta": np.deg2rad(15)
    }

    steering_angles = np.deg2rad([-30, -15, 0, 15, 30])
    speeds = [10, -10]

    came_from = {}

    while open_set:
        _, current_cost, (x, y, theta), parent = heapq.heappop(open_set)

        # Discrétisation
        key = (
            int(x // resolutions["x"]),
            int(y // resolutions["y"]),
            int(theta // resolutions["theta"]) % int(2 * np.pi // resolutions["theta"])
        )

        if key in closed_set:
            continue
        closed_set.add(key)
        came_from[key] = (x, y, theta, parent)

        if heuristic(x, y, xg, yg) < 10:
            print("🚗 Chemin trouvé")
            path = []
            while parent is not None:
                path.append((x, y, theta))
                _, _, (x, y, theta), parent = parent
            path.append((x0, y0, theta0))
            return path[::-1]

        for delta in steering_angles:
            for speed in speeds:
                x_new, y_new, theta_new = move_car(x, y, theta, speed, delta, dt=1.0)

                if not is_in_bounds(x_new, y_new, map_shape):
                    continue

                _, collision = check_circular_collision_distance_2_circles(dist_map, x_new, y_new, theta_new)
                if collision:
                    continue

                new_cost = current_cost + 1
                new_state = (x_new, y_new, theta_new)
                heapq.heappush(open_set, (new_cost + heuristic(x_new, y_new, xg, yg), new_cost, new_state, (_, current_cost, (x, y, theta), parent)))

    print("❌ Échec : aucun chemin trouvé.")
    return None


path = hybrid_a_star(start=(150, 200, 0),
                     goal=(170, 150, np.pi/2),
                     dist_map=dist_map,
                     map_shape=map_shape)

print(path)
