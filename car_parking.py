import numpy as np
import cv2
import math

class Car:
    def __init__(self, x, y, theta, L=13, car_width=9, car_height=17):
        self.x = x
        self.y = y
        self.theta = theta  # en degrés
        self.L = L  # empattement (distance entre essieux)
        self.car_width = car_width
        self.car_height = car_height

        # Création d'une image de la voiture autour de l'essieu arrière
        self.rear_axle_offset_y = int(0.75 * self.car_height)
        self.img_height = 2 * self.rear_axle_offset_y
        self.img_width = 2 * (self.car_width // 2 + 1)  # +1 pour éviter les problèmes avec les dimensions paires

        self._create_car_image()

    def _create_car_image(self):
        """Crée l'image RGBA de la voiture (fond transparent)"""
        self.car_img = np.zeros((self.img_height, self.img_width, 4), dtype=np.uint8)

        # Dessiner le corps de la voiture
        top_left = (self.img_width // 2 - self.car_width // 2,
                    self.rear_axle_offset_y - self.car_height)
        bottom_right = (self.img_width // 2 + self.car_width // 2,
                        self.rear_axle_offset_y)

        cv2.rectangle(self.car_img, top_left, bottom_right, (0, 0, 255, 255), thickness=-1)

        # Marquer le centre de rotation (essieu arrière)
        center_x = self.img_width // 2
        center_y = self.rear_axle_offset_y
        cv2.circle(self.car_img, (center_x, center_y), 2, (255, 0, 0, 255), -1)

    def apply_control(self, v, phi_deg, dt=1.0):
        """Applique un contrôle cinématique (modèle vélo simple)
        
        Args:
            v: vitesse (pixels/frame)
            phi_deg: angle du volant en degrés
            dt: pas de temps
        """
        theta_rad = math.radians(self.theta)
        phi_rad = math.radians(phi_deg)

        dx = v * math.cos(theta_rad) * dt
        dy = -v * math.sin(theta_rad) * dt  # Axe y vers le bas
        
        self.x += dx
        self.y += dy
        self.theta += math.degrees((v / self.L) * math.tan(phi_rad) * dt)
        self.theta %= 360  # Normalisation à [0, 360[

    def draw_on_map(self, map_img):
        """Dessine la voiture sur la carte avec OpenCV"""
        # Rotation de la voiture
        center = (self.car_img.shape[1] // 2, self.car_img.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -self.theta, 1.0)
        rotated_car = cv2.warpAffine(self.car_img, rot_mat,
                                (self.car_img.shape[1], self.car_img.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))

        # Convertir la carte en BGRA si nécessaire
        if map_img.shape[2] == 3:
            map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2BGRA)

        # Position de dessin
        pos_x = int(self.x - rotated_car.shape[1] / 2)
        pos_y = int(self.y - rotated_car.shape[0] / 2)

        # Vérifier les bords
        h, w = rotated_car.shape[:2]
        if (pos_x < 0 or pos_y < 0 or 
            pos_x + w > map_img.shape[1] or 
            pos_y + h > map_img.shape[0]):
            return map_img  # Ne pas dessiner si en dehors

        # Copie de la carte pour modification
        map_with_car = map_img.copy()
        roi = map_with_car[pos_y:pos_y+h, pos_x:pos_x+w]

        # Extraction du canal alpha et préparation pour broadcasting
        car_alpha = rotated_car[:, :, 3] / 255.0  # Forme (h,w)
        car_rgb = rotated_car[:, :, :3]          # Forme (h,w,3)
        
        # Fusion alpha
        for c in range(3):
            roi[:, :, c] = (car_alpha * car_rgb[:, :, c] + 
                        (1 - car_alpha) * roi[:, :, c]).astype(np.uint8)

        return map_with_car

def load_map(map_path):
    """Charge une carte existante depuis un fichier"""
    map_img = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
    
    if map_img is None:
        raise FileNotFoundError(f"Impossible de charger la carte à partir de {map_path}")
    
    # Convertir en BGRA si nécessaire
    if map_img.shape[2] == 3:
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2BGRA)
    
    return map_img

# Charger la carte existante (remplacez par votre chemin)
try:
    map_img = load_map("mapLittle .png")  # Format supporté: PNG, JPG, etc.
except FileNotFoundError as e:
    print(e)
    # Fallback: créer une carte vide si la carte n'est pas trouvée
    map_img = np.zeros((500, 500, 3), dtype=np.uint8)
    map_img.fill(255)  # Fond blanc
    print("Utilisation d'une carte vierge par défaut")

# Création de la voiture (position initiale au centre de la carte)
car = Car(x=map_img.shape[1]//2, y=map_img.shape[0]//2, theta=0)

# Configuration de la fenêtre
cv2.namedWindow("Simulation Parking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Simulation Parking", 800, 600)

# Simulation de mouvement
while True:
    # Appliquer les commandes de contrôle
    car.apply_control(v=2, phi_deg=5, dt=1.0)
    
    # Dessiner sur une copie de la carte
    current_map = car.draw_on_map(map_img.copy())
    
    # Afficher
    cv2.imshow("Simulation Parking", current_map)
    
    # Contrôles clavier:
    key = cv2.waitKey(30)
    if key == 27:  # ESC pour quitter
        break
    elif key == ord('r'):  # R pour réinitialiser
        car = Car(x=map_img.shape[1]//2, y=map_img.shape[0]//2, theta=0)

cv2.destroyAllWindows()