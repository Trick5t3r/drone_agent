import sys
import random
import math
from pathlib import Path
from typing import Optional, List, Tuple
from enum import Enum
import numpy as np
import cv2
import arcade
import heapq

from spg.playground import Playground
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.entities.rescue_center import RescueCenter
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, circular_mean, clamp
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.path import Path

# =======================
# CONSTANTES DE CONFIGURATION
# =======================
GRID_EVERY_N = 3
LIDAR_CLIP_DIST = 40.0
VAL_ZONE_VIDE = -0.602
VAL_ZONE_OBSTACLE = 2.0
VAL_ZONE_LIBRE = -4.0
VAL_INITIALE = 0.0
CLIP_MIN = -40
CLIP_MAX = 40
VAL_RESCUE_ZONE = 100.0

# =======================
# CLASSE : OccupancyGrid
# =======================
class OccupancyGrid(Grid):
    """Grille d'occupation simplifiée"""

    def __init__(self, world_size, resolution: float, lidar_sensor):
        super().__init__(size_area_world=world_size, resolution=resolution)
        self.world_size = world_size
        self.resolution = resolution
        self.lidar_sensor = lidar_sensor

        self.grid_width = int(self.world_size[0] / self.resolution + 0.5)
        self.grid_height = int(self.world_size[1] / self.resolution + 0.5)

        self.grid = np.zeros((self.grid_width, self.grid_height))
        self.zoomed_grid = np.empty((self.grid_width, self.grid_height))

    def update_grid(self, pose: Pose, semantic_detections: Optional[List] = None):
        # Récupération et sous-échantillonnage des mesures lidar
        lidar_distances = self.lidar_sensor.get_sensor_values()[::GRID_EVERY_N].copy()
        lidar_angles = self.lidar_sensor.ray_angles[::GRID_EVERY_N].copy()

        # Calcul des directions des rayons
        cos_angles = np.cos(lidar_angles + pose.orientation)
        sin_angles = np.sin(lidar_angles + pose.orientation)
        lidar_max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # Calcul des points "vides" (sans obstacle détecté)
        distances_vides = np.maximum(lidar_distances - LIDAR_CLIP_DIST, 0.0)
        distances_vides_clip = np.minimum(distances_vides, lidar_max_range)
        empty_x = pose.position[0] + distances_vides_clip * cos_angles
        empty_y = pose.position[1] + distances_vides_clip * sin_angles

        for x, y in zip(empty_x, empty_y):
            self.add_value_along_line(pose.position[0], pose.position[1], x, y, VAL_ZONE_VIDE)

        # Calcul des points correspondant aux obstacles détectés
        collision_mask = lidar_distances < lidar_max_range
        obs_x = pose.position[0] + lidar_distances * cos_angles
        obs_y = pose.position[1] + lidar_distances * sin_angles
        obs_x = obs_x[collision_mask]
        obs_y = obs_y[collision_mask]
        self.add_points(obs_x, obs_y, VAL_ZONE_OBSTACLE)

        # Marquage de la position actuelle comme zone libre
        self.add_points(pose.position[0], pose.position[1], VAL_ZONE_LIBRE)

        # Mise à jour de la rescue zone via les détections sémantiques (si fournies)
        if semantic_detections is not None:
            for detection in semantic_detections:
                if detection.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    rescue_x = pose.position[0] + detection.distance * math.cos(detection.angle + pose.orientation)
                    rescue_y = pose.position[1] + detection.distance * math.sin(detection.angle + pose.orientation)
                    self.add_points(rescue_x, rescue_y, VAL_ZONE_LIBRE)

        # Mise à jour de la rescue zone via les détections sémantiques (si fournies)
        if semantic_detections is not None:
            for detection in semantic_detections:
                if detection.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    rescue_x = pose.position[0] + detection.distance * math.cos(detection.angle + pose.orientation)
                    rescue_y = pose.position[1] + detection.distance * math.sin(detection.angle + pose.orientation)
                    self.add_points(rescue_x, rescue_y, VAL_RESCUE_ZONE)

        # Clip de la grille tout en préservant les zones rescue
        clipped = np.clip(self.grid, CLIP_MIN, CLIP_MAX)
        self.grid = np.where(self.grid == VAL_RESCUE_ZONE, VAL_RESCUE_ZONE, clipped)

        # Création d'une version "zoomée" de la grille
        self.zoomed_grid = self.grid.copy()
        new_size = (int(self.world_size[1] * 0.5), int(self.world_size[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_size, interpolation=cv2.INTER_NEAREST)


# =======================
# ALGORITHME A* POUR LE PATHFINDING
# =======================
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def a_star(grid_matrix, start: Tuple[int, int], goal: Tuple[int, int], mode='search'):
    """
    Algorithme A* fusionné.
      - grid_matrix : matrice de la grille d'occupation.
      - start : cellule de départ.
      - goal : cellule cible.
      - mode : "search" arrête sur une cellule de valeur 0, sinon sur le goal exact.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if mode == 'search':
            if grid_matrix[current[0], current[1]] == 0:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
        elif mode == 'rescue_zone':
            if grid_matrix[current[0], current[1]] == VAL_RESCUE_ZONE:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
        else:
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid_matrix.shape[0] and 0 <= neighbor[1] < grid_matrix.shape[1]:
                # Vérification du passage selon la valeur de la cellule
                if grid_matrix[neighbor[0], neighbor[1]] >= VAL_ZONE_OBSTACLE:
                    if mode != "rescue_zone" or grid_matrix[neighbor[0], neighbor[1]] != VAL_RESCUE_ZONE:
                        continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None


# =======================
# CLASSE : MyDroneEval
# =======================
class MyDroneEval(DroneAbstract):
    class Activity(Enum):
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, display_lidar_graph=False, **kwargs)
        self.state = MyDroneEval.Activity.SEARCHING_WOUNDED
        self.straight_counter = 0
        self.target_turn_angle = 0
        self.is_turning = False
        self.iteration = 0
        self.estimated_pose = Pose()
        self.prev_angle_error = 0.0
        self.command_counter = 0
        self.direction_change_interval = 50
        self.Kp = 9.0
        self.Kd = 0.6
        self.forward_speed = 1.0

        self.explored_reward = 0

        resolution = 8
        self.grid = OccupancyGrid(world_size=self.size_area, resolution=resolution, lidar_sensor=self.lidar())
        self.path_done = Path()
        self.last_known_inside_area = None

    def define_message_for_all(self):
        msg_data = (self.identifier,
                    (self.measured_gps_position(),
                     self.measured_compass_angle()), 
                     self.grid.grid)
        return msg_data

    def process_communication_sensor(self):
        grid_values = [self.grid.grid]

        if self.communicator:
            received_messages = self.communicator.received_messages
            for msg in received_messages:
                grid_values.append(msg[2])

        # Empiler toutes les matrices dans un tableau 3D de forme (n, rows, cols)
        stacked = np.stack(grid_values, axis=0)
        
        # Pour chaque élément, on détermine s'il existe une valeur > 0 parmi les matrices
        positive_mask = stacked > 0  # masque booléen de même forme que stacked
        has_positive = np.any(positive_mask, axis=0)  # booléen pour chaque (i, j)

        # Pour les positions avec au moins une valeur positive, on prend le maximum positif.
        # On remplace les valeurs non-positives par -inf pour ne pas fausser le max.
        pos_max = np.max(np.where(positive_mask, stacked, -np.inf), axis=0)
        
        # Pour les positions sans valeur positive, on prend la valeur minimale parmi toutes les matrices.
        overall_min = np.min(stacked, axis=0)
        
        # Combiner les deux résultats en fonction de la présence de valeurs positives
        self.grid.grid = np.where(has_positive, pos_max, overall_min)

            
                

    def control(self):
        # Initialisation de la commande par défaut
        print("reward exploration:", self.explored_reward)
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Traitement du capteur sémantique
        found_wounded, found_rescue, semantic_command = self.process_semantic_sensor()

        # Mise à jour de l'état en fonction des détections et de l'état du grappin
        if self.state == MyDroneEval.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = MyDroneEval.Activity.GRASPING_WOUNDED
        elif self.state == MyDroneEval.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = MyDroneEval.Activity.SEARCHING_RESCUE_CENTER
        elif self.state == MyDroneEval.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = MyDroneEval.Activity.SEARCHING_WOUNDED
        elif self.state == MyDroneEval.Activity.SEARCHING_RESCUE_CENTER and found_rescue:
            self.state = MyDroneEval.Activity.DROPPING_AT_RESCUE_CENTER
        elif self.state == MyDroneEval.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.state = MyDroneEval.Activity.SEARCHING_WOUNDED
        elif self.state == MyDroneEval.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue:
            self.state = MyDroneEval.Activity.SEARCHING_RESCUE_CENTER

        print(f"État: {self.state.name}, peut saisir: {self.base.grasper.can_grasp}, entités saisies: {self.base.grasper.grasped_entities}")

        if self.state == MyDroneEval.Activity.SEARCHING_WOUNDED:
            command = self.control_explore()
            command["grasper"] = 0
        elif self.state == MyDroneEval.Activity.GRASPING_WOUNDED:
            command = semantic_command
            command["grasper"] = 1
        elif self.state == MyDroneEval.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_rescue_zone()
            command["grasper"] = 1
        elif self.state == MyDroneEval.Activity.DROPPING_AT_RESCUE_CENTER:
            command = semantic_command
            command["grasper"] = 1
        else:
            command = self.control_return_area()
            command["grasper"] = 1

        self.iteration += 1
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()), self.measured_compass_angle())
        semantic_detections = self.semantic_values() or []
        self.grid.update_grid(pose=self.estimated_pose, semantic_detections=semantic_detections)

        if self.is_inside_return_area:
            self.last_known_inside_area = self.estimated_pose

        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid, self.estimated_pose, title="Grille d'occupation")
            self.grid.display(self.grid.zoomed_grid, self.estimated_pose, title="Grille zoomée")

        return command

    def draw_bottom_layer(self):
        self.draw_path(path=self.path_done, color=(255, 0, 255))

    def draw_path(self, path: Path, color: Tuple[int, int, int]):
        previous_point = None
        for idx in range(path.length()):
            pose_point = path.get(idx)
            current_point = pose_point.position + self._half_size_array
            if previous_point is not None:
                arcade.draw_line(float(previous_point[0]), float(previous_point[1]),
                                 float(current_point[0]), float(current_point[1]), color)
            previous_point = current_point

    # Méthode commune pour calculer une commande PD vers une cible donnée (en coordonnées de grille)
    def compute_pd_command(self, target_grid: Tuple[int, int]) -> dict:
        target_world = np.asarray(self.grid._conv_grid_to_world(target_grid[0], target_grid[1]))
        # On enregistre la position cible dans le chemin parcouru
        self.path_done.append(Pose(target_world, self.measured_compass_angle()))
        current_pos = np.array(self.estimated_pose.position)
        direction = target_world - current_pos
        desired_angle = math.atan2(direction[1], direction[0])
        angle_error = normalize_angle(desired_angle - self.measured_compass_angle())
        derivative_error = normalize_angle(angle_error - self.prev_angle_error)
        rotation = clamp(self.Kp * angle_error + self.Kd * derivative_error, -1.0, 1.0)
        forward = self.forward_speed * (1 - abs(angle_error) / math.pi)
        self.prev_angle_error = angle_error
        return {"forward": forward, "rotation": rotation}

    def control_explore(self):
        self.command_counter += 1
        grid_matrix = self.grid.grid
        free_indices = np.argwhere(grid_matrix == VAL_INITIALE)
        if free_indices.size == 0:
            print("Aucune zone libre trouvée pour l'exploration.")
            return {"forward": 0.0, "rotation": 0.0}

        barycenter = tuple(map(int, free_indices.mean(axis=0)))
        current_grid = self.grid._conv_world_to_grid(int(self.estimated_pose.position[0]),
                                                     int(self.estimated_pose.position[1]))
        path = a_star(grid_matrix, current_grid, barycenter, mode='search')
        if not path or len(path) < 2:
            print("Aucun chemin trouvé vers le barycentre.")
            return self.control_random()

        next_cell = path[min(2, len(path)-1)]
        return self.compute_pd_command(next_cell)

    def process_lidar_sensor(self) -> bool:
        lidar_vals = self.lidar_values()
        if lidar_vals is None:
            return False
        return min(lidar_vals) < 40

    def control_random(self):
        cmd_forward = {"forward": 0.5, "rotation": 0.0}
        cmd_turn = {"forward": 0.0, "rotation": 1.0}
        collided = self.process_lidar_sensor()
        self.straight_counter += 1

        if collided and (not self.is_turning) and self.straight_counter > 100:
            self.is_turning = True
            self.target_turn_angle = random.uniform(-math.pi, math.pi)

        angle_diff = normalize_angle(self.target_turn_angle - self.measured_compass_angle())
        if self.is_turning and abs(angle_diff) < 0.2:
            self.is_turning = False
            self.straight_counter = 0

        return cmd_turn if self.is_turning else cmd_forward

    def control_rescue_zone(self):
        target_cell = self.grid._conv_world_to_grid(int(self.estimated_pose.position[0]),
                                                       int(self.estimated_pose.position[1]))
        grid_matrix = self.grid.grid
        rescue_cells = np.argwhere(grid_matrix == VAL_RESCUE_ZONE)
        if rescue_cells.size == 0:
            print("Aucune rescue zone trouvée dans la grille.")
            if self.is_inside_return_area or self.last_known_inside_area is None:
                return self.control_explore()
            target_cell = self.grid._conv_world_to_grid(int(self.last_known_inside_area.position[0]),
                                                         int(self.last_known_inside_area.position[1]))
        else:
            distances = np.linalg.norm(rescue_cells - np.array(target_cell), axis=1)
            target_cell = tuple(rescue_cells[np.argmin(distances)])
            print("Cible (rescue zone) en grille:", target_cell)
        current_grid = self.grid._conv_world_to_grid(int(self.estimated_pose.position[0]),
                                                     int(self.estimated_pose.position[1]))

        path = a_star(grid_matrix, current_grid, target_cell, mode='rescue_zone')
        if not path or len(path) < 2:
            print("Aucun chemin trouvé vers la rescue zone.")
            return self.control_random()
        next_cell = path[min(2, len(path)-1)]
        return self.compute_pd_command(next_cell)

    def control_return_area(self):
        if self.is_inside_return_area or self.last_known_inside_area is None:
            return self.control_random()

        current_grid = self.grid._conv_world_to_grid(int(self.estimated_pose.position[0]),
                                                     int(self.estimated_pose.position[1]))
        target_grid = self.grid._conv_world_to_grid(int(self.last_known_inside_area.position[0]),
                                                    int(self.last_known_inside_area.position[1]))
        grid_matrix = self.grid.grid
        print("Retour: position actuelle", current_grid, "taille grille", grid_matrix.shape)
        path = a_star(grid_matrix, current_grid, target_grid)
        if not path or len(path) < 2:
            print("Aucun chemin trouvé vers la zone de retour.")
            return self.control_random()
        next_cell = path[min(2, len(path)-1)]
        return self.compute_pd_command(next_cell)

    def process_semantic_sensor(self):
        cmd = {"forward": 0.5, "lateral": 0.0, "rotation": 0.0}
        angular_max = 1.0
        detections = self.semantic_values()
        best_angle = 0
        found_wounded = False

        # Traitement des personnes blessées
        if self.state in [MyDroneEval.Activity.SEARCHING_WOUNDED, MyDroneEval.Activity.GRASPING_WOUNDED] and detections:
            wounded_scores = []
            for det in detections:
                if det.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not det.grasped:
                    found_wounded = True
                    score = (det.angle ** 2) + (det.distance ** 2 / 1e5)
                    wounded_scores.append((score, det.angle, det.distance))
            if wounded_scores:
                best_score = min(wounded_scores, key=lambda x: x[0])
                best_angle = best_score[1]

        # Traitement des rescue centers
        found_rescue = False
        near_rescue = False
        rescue_angles = []
        if self.state in [MyDroneEval.Activity.SEARCHING_RESCUE_CENTER, MyDroneEval.Activity.DROPPING_AT_RESCUE_CENTER] and detections:
            for det in detections:
                if det.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue = True
                    rescue_angles.append(det.angle)
                    near_rescue = det.distance < 30
            if found_rescue:
                best_angle = circular_mean(np.array(rescue_angles))

        if found_rescue or found_wounded:
            rotation_val = max(min(2.0 * best_angle, 1.0), -1.0) * angular_max
            cmd["rotation"] = rotation_val
            if abs(rotation_val) == 1:
                cmd["forward"] = 0.2

        if found_rescue and near_rescue:
            cmd["forward"] = 0.0
            cmd["rotation"] = -1.0

        return found_wounded, found_rescue, cmd
