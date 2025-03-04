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

# Import de PyTorch pour l'agent DDPG
import torch
import torch.nn as nn
import torch.optim as optim

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

###########################################
# PARTIE DDPG : Définition des réseaux et agent
###########################################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Utilisation de tanh pour obtenir des sorties dans [-1,1]
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, batch_size=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = ReplayBuffer()

        # Paramètre pour l'exploration (bruit gaussien)
        self.noise_std = 0.2

    def select_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise
        # Pour la vitesse forward, on peut la forcer dans [0,1] en transformant la sortie
        action[0] = np.clip((action[0] + 1) / 2, 0, 1)  # transformation de [-1,1] vers [0,1]
        action[1] = np.clip(action[1], -1, 1)
        return action

    def push(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).reshape(-1,1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).reshape(-1,1).to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_actions[:,0] = (next_actions[:,0] + 1) / 2  # transformation pour la vitesse
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_actions = self.actor(states)
        # On transforme la première composante en [0,1]
        actor_actions_transformed = actor_actions.clone()
        actor_actions_transformed[:,0] = (actor_actions_transformed[:,0] + 1) / 2
        actor_loss = -self.critic(states, actor_actions_transformed).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update des réseaux cibles
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
    
    def save(self, path):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'memory': self.memory.buffer,  # Optionnel : sauvegarde du replay buffer
            'noise_std': self.noise_std,
            'gamma': self.gamma,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'state_dim': self.state_dim,  # Configuration : dimension de l'état
            'action_dim': self.action_dim  # Configuration : dimension de l'action
        }
        torch.save(checkpoint, path)


    def load(self, path):
        """
        agent = DDPGAgent(state_dim, action_dim)
        agent.load("chemin/vers/le_fichier_checkpoint.pth")
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Charger la configuration (dimensions de l'état et de l'action)
            self.state_dim = checkpoint.get('state_dim', self.state_dim)
            self.action_dim = checkpoint.get('action_dim', self.action_dim)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.memory.buffer = checkpoint['memory']  # Recharge du replay buffer
            self.noise_std = checkpoint.get('noise_std', self.noise_std)
            self.gamma = checkpoint.get('gamma', self.gamma)
            self.tau = checkpoint.get('tau', self.tau)
            self.batch_size = checkpoint.get('batch_size', self.batch_size)
            return True
        except FileNotFoundError:
            print("file not found")
            return False





###########################################
# FIN PARTIE DDPG
###########################################

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
        self._old_explored_reward = 0

        resolution = 8
        self.grid = OccupancyGrid(world_size=self.size_area, resolution=resolution, lidar_sensor=self.lidar())
        self.path_done = Path()
        self.last_known_inside_area = None

        # Initialisation de l'agent DDPG pour le mode exploration.
        grid_shape = self.grid.grid.shape  # (largeur, hauteur)
        state_dim = 2 + 1 + grid_shape[0] * grid_shape[1]  # position (2), angle (1) et grille aplatie
        action_dim = 2  # forward et rotation
        self.ddpg_agent = DDPGAgent(state_dim, action_dim)
        self.last_explore_state = None
        self.last_explore_action = None

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
        stacked = np.stack(grid_values, axis=0)
        positive_mask = stacked > 0
        has_positive = np.any(positive_mask, axis=0)
        pos_max = np.max(np.where(positive_mask, stacked, -np.inf), axis=0)
        overall_min = np.min(stacked, axis=0)
        self.grid.grid = np.where(has_positive, pos_max, overall_min)

    def control(self):
        print("reward exploration:", self.explored_reward, "last :", self._old_explored_reward, "diff:", self.explored_reward -self._old_explored_reward)
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        found_wounded, found_rescue, semantic_command = self.process_semantic_sensor()

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

        self._old_explored_reward = self.explored_reward

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

    def compute_pd_command(self, target_grid: Tuple[int, int]) -> dict:
        target_world = np.asarray(self.grid._conv_grid_to_world(target_grid[0], target_grid[1]))
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
        # Construction de l'état : position (2), angle (1) et grille aplatie
        grid_matrix = self.grid.grid
        state = np.concatenate([
            np.array(self.estimated_pose.position).flatten(),
            np.array([self.measured_compass_angle()]),
            grid_matrix.flatten()
        ])
        # Si un état précédent existe, mettre à jour l'agent avec la transition
        diff = self.explored_reward - self._old_explored_reward
        if self.last_explore_state is not None:
            self.ddpg_agent.push(self.last_explore_state, self.last_explore_action, 
                                   diff*10000, state, False)
            self.ddpg_agent.update()
        
        action = self.ddpg_agent.select_action(state)
        # Si la différence est nulle, on force une action d'exploration aléatoire
        if diff == 0:
            # On peut définir une action aléatoire dans l'intervalle désiré :
            action += np.array([random.uniform(0, 1), random.uniform(-1, 1)])
            action /= 2

        self.last_explore_state = state
        self.last_explore_action = action
        # L'action est un vecteur [forward, rotation]
        cmd = {"forward": float(action[0]), "rotation": float(action[1])}
        return cmd

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
