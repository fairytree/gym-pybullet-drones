import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ExploreAviary(BaseRLAviary):
    """Single agent RL problem: Explore a predefined area."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 incentive_options: dict=None
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.grid_size = 0.2  # discretization step in meters
        self.EPISODE_LEN_SEC = 120
        self.incentive_options = incentive_options

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         incentive_options=incentive_options
                         )

        # Each obstacle: (center_position, half_extents)
        self.obstacles = [
            (np.array(obs["position"]), np.array(obs["size"]) / 2)
            for obs in self.obstacles_info
        ]
        self.last_waypoint = np.array([0.0, 0.0, 0.0]) 
        
        # Precompute grid dimensions
        self.nx = int(round((self.bounds[0, 1] - self.bounds[0, 0]) / self.grid_size)) + 1
        self.ny = int(round((self.bounds[1, 1] - self.bounds[1, 0]) / self.grid_size)) + 1
        self.nz = int(round((self.bounds[2, 1] - self.bounds[2, 0]) / self.grid_size)) + 1
        self.total_voxels = self.nx * self.ny * self.nz

        # Visited voxels as 1D boolean array
        self.visited_mask = np.zeros(self.total_voxels, dtype=bool)  
        self.drawn_mask = np.zeros_like(self.visited_mask)
        self.visited = set()

        
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.
        """

        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        idx3d, out_of_bounds = self.pos_to_idx3d(current_pos)
        idx1d = self.idx3d_to_1d(idx3d)

        reward = 0.0

        # reward for visiting new voxel
        if self.incentive_options.get("new_voxel_reward", True):
            if not self.visited_mask[idx1d] and not out_of_bounds:
                reward += 10.0
                self.visited_mask[idx1d] = True
                self.visited.add(idx3d)
            else:
                reward -= 0.1  # penalty for revisiting, try smaller penalty if too harsh

        # Penalty for leaving bounds
        if self.incentive_options.get("out_of_boundary_penalty", True):
            if np.any(current_pos < self.bounds[:,0]) or np.any(current_pos > self.bounds[:,1]):
                reward -= 0.1

        # direction-change penalty
        if self.incentive_options.get("change_direction_penalty", True):
            reward += self._direction_penalty(current_pos, self.last_waypoint, self.current_waypoint)
            self.last_waypoint = self.current_waypoint.copy()

        # Penalty for collisions with obstacles (boxes)
        if self.incentive_options.get("collision_penalty", True) and self._checkCollision(state[0:3]):
            reward -= 100

        # time penalty
        if self.incentive_options.get("time_penalty", True):
            reward -= 0.0001

        return reward

    ################################################################################
    
    def _direction_penalty(self, current_pos, last_waypoint, current_waypoint):
        v_prev = last_waypoint - current_pos
        v_next = current_waypoint - current_pos

        # avoid zero vectors
        if np.linalg.norm(v_prev) < 1e-6 or np.linalg.norm(v_next) < 1e-6:
            return 0.0

        # dot product
        dot = np.dot(v_prev, v_next)

        if dot <= 0:
            return -1  # penalty for reversal
        else:
            return 0.0


    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        return False  # exploration continues until truncated

        
    ################################################################################
    

    def _computeTruncated(self):
        """Truncate episode if out-of-bounds, collision, or time out."""
        state = self._getDroneStateVector(0)
        x, y, z = state[0:3]

        # Check if drone is out of bounds
        buffer = 1.0
        out_of_bounds = (x < self.bounds[0,0] - buffer) or (x > self.bounds[0,1] + buffer) \
                        or (y < self.bounds[1,0] - buffer) or (y > self.bounds[1,1] + buffer) \
                        or (z < self.bounds[2,0] - buffer) or (z > self.bounds[2,1] + buffer)

        # Check if drone is tilted too much
        tilted = abs(state[7]) > 0.4 or abs(state[8]) > 0.4

        # Check collision with obstacles (boxes)
        collision = self._checkCollision(state[0:3])
        
        # Check timeout
        timeout = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

        if (out_of_bounds):
            print("Truncated: Out of bounds")
        if (tilted):
            print("Truncated: Tilted")
        if (collision):
            print("Truncated: Collision")
        if (timeout):
            print("Truncated: Timeout")
        return out_of_bounds or tilted or collision or timeout

    ################################################################################

    def _checkCollision(self, pos):
        for obs_pos, half_extents in self.obstacles:
            obs_min = obs_pos - half_extents - 0.1
            obs_max = obs_pos + half_extents + 0.1
            if np.all(pos > obs_min) and np.all(pos < obs_max):
                return True
        return False

    ################################################################################

    def _nearest_unexplored_voxel(self, max_steps=5):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_idx3d, out_of_bounds = self.pos_to_idx3d(current_pos)

        # incremental search
        if not out_of_bounds:
            for step in range(1, max_steps+1):
                for dx in range(-step, step+1):
                    for dy in range(-step, step+1):
                        for dz in range(-step, step+1):
                            candidate = (current_idx3d[0]+dx, current_idx3d[1]+dy, current_idx3d[2]+dz)
                            if 0 <= candidate[0] < self.nx and 0 <= candidate[1] < self.ny and 0 <= candidate[2] < self.nz:
                                idx1d = self.idx3d_to_1d(candidate)
                                if not self.visited_mask[idx1d]:
                                    target_pos = np.array(candidate) * self.grid_size + self.bounds[:,0]
                                    # target_pos = target_pos - current_pos # output relative position
                                    # Check collision along straight line
                                    if not self._checkCollision(target_pos):
                                        return np.linalg.norm(target_pos), target_pos

        # fallback: return first unvisited voxel
        unvisited = np.where(~self.visited_mask)[0]
        for step in range(min(max_steps, len(unvisited))):
            candidate = unvisited[step]
            target_pos = self.idx1d_to_pos(candidate)
            if not self._checkCollision(target_pos):
                # target_pos = target_pos - current_pos # output relative position
                return np.linalg.norm(target_pos), target_pos

        return 0.0, np.zeros(3)
                        

    ################################################################################

    def _exploration_percentage(self):
        return np.sum(self.visited_mask) / self.total_voxels

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def visualize_visited_voxels(self):
        color = [0.0, 1.0, 0.0, 0.3]
        visited_indices = np.where(self.visited_mask)[0]
        for idx in visited_indices:
            voxel_center = self.idx1d_to_pos(idx) + self.grid_size/2
            half_extents = [self.grid_size/2]*3
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=collision_shape,
                              baseVisualShapeIndex=visual_shape,
                              basePosition=voxel_center,
                              physicsClientId=self.CLIENT)

    def visualize_trajectory(self):
        if not hasattr(self, 'trajectory'):
            self.trajectory = []
        state = self._getDroneStateVector(0)
        self.trajectory.append(state[0:3])
        if len(self.trajectory) > 1:
            p.addUserDebugLine(self.trajectory[-2], self.trajectory[-1], [1,0,0], 2, lifeTime=0)


  ################################################################################

    def pos_to_idx3d(self, pos):
        """Convert 3D position to voxel indices (i,j,k). Return out_of_bounds flag."""
        i = int(np.round((pos[0] - self.bounds[0,0]) / self.grid_size))
        j = int(np.round((pos[1] - self.bounds[1,0]) / self.grid_size))
        k = int(np.round((pos[2] - self.bounds[2,0]) / self.grid_size))
        
        out_of_bounds = False
        if i < 0 or i >= self.nx: out_of_bounds = True
        if j < 0 or j >= self.ny: out_of_bounds = True
        if k < 0 or k >= self.nz: out_of_bounds = True

        # Clamp to valid range (to avoid indexing errors)
        i = np.clip(i, 0, self.nx - 1)
        j = np.clip(j, 0, self.ny - 1)
        k = np.clip(k, 0, self.nz - 1)
        
        return (i, j, k), out_of_bounds


    def idx3d_to_1d(self, idx3d):
        """Convert 3D voxel index to 1D index."""
        i, j, k = idx3d
        return i * (self.ny * self.nz) + j * self.nz + k

    def idx1d_to_pos(self, idx1d):
        """Convert 1D index back to voxel center position."""
        i = idx1d // (self.ny * self.nz)
        j = (idx1d % (self.ny * self.nz)) // self.nz
        k = idx1d % self.nz
        pos = np.array([i, j, k]) * self.grid_size + self.bounds[:,0]
        return pos