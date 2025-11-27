import numpy as np

from gym_pybullet_drones.envs.ParticleFilter import ParticleFilter as pf


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
                 act: ActionType=ActionType.RPM
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
        self.visited = set()
        self.grid_size = 0.2  # discretization step in meters

        self.EPISODE_LEN_SEC = 40

        #setting up distances
        self.closest_distance = np.inf
        self.last_distance = np.inf
        
        
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
                         act=act
                         )
        

        # Each obstacle: (center_position, half_extents)
        self.obstacles = [
            (np.array(obs["position"]), np.array(obs["size"]) / 2)
            for obs in self.obstacles_info
        ]
        self.last_waypoint = np.array([0.0, 0.0, 0.0])
        

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.
        """

        state = self._getDroneStateVector(0)
        pos = tuple(np.round(state[0:3] / self.grid_size).astype(int))
        reward = 0.0
        alpha = 0.1
        did_sense=self.action_buffer[-1][0,-1] > 0
        
        #penalty for existing
        reward += -0.01

        # add reward for reaching the target (Need to add distance_to_target info as state)
        distance_to_target = np.linalg.norm(state[0:3] - self.emmit_target)
        if distance_to_target < 0.2:
            reward += 1000.0
        
        # reward for reducing KL divergence (and not being at the target)
        if did_sense and distance_to_target >= 0.2:
            #penalty for sensing
            reward += -0.5
            distance_with_error=distance_to_target + np.random.normal(loc=0,scale=self.measurement_sd)
            KL_reward=self.filter.KL_divergence(state[0:3],distance_with_error)
            reward += alpha*KL_reward
            self.filter.predict()
        
        # penalty to stop the drone flying above the walls
        if state[2]>0.5:
            reward -= state[2]

        # Penalty for leaving bounds
        if np.any(state[0:3] < self.bounds[:,0]) or np.any(state[0:3] > self.bounds[:,1]):
            reward -= 0.1

        # direction-change penalty
        reward += self._direction_penalty(state[0:3], self.last_waypoint, self.current_waypoint)
        self.last_waypoint = self.current_waypoint.copy()

        # Penalty for collisions with obstacles (boxes)
        for obs_pos, half_extents in self.obstacles:
            obs_min = obs_pos - half_extents - 0.1  # small safety buffer
            obs_max = obs_pos + half_extents + 0.1
            if np.all(state[0:3] > obs_min) and np.all(state[0:3] < obs_max):
                reward -= 100

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

        # check if drone is at the target
        state = self._getDroneStateVector(0)
        distance_to_target = np.linalg.norm(state[0:3] - self.emmit_target)
        if distance_to_target < 0.2:
            done=True
        else: done = False

        return done  # exploration continues until truncated

        
    ################################################################################
    

    def _computeTruncated(self):
        """Truncate episode if out-of-bounds, collision, or time out."""
        state = self._getDroneStateVector(0)
        x, y, z = state[0:3]

        # Check if drone is out of bounds
        buffer = 0.5
        out_of_bounds = (x < self.bounds[0,0] - buffer) or (x > self.bounds[0,1] + buffer) \
                        or (y < self.bounds[1,0] - buffer) or (y > self.bounds[1,1] + buffer) \
                        or (z < self.bounds[2,0] - buffer) or (z > self.bounds[2,1] + buffer)

        # Check if drone is tilted too much
        tilted = abs(state[7]) > 0.4 or abs(state[8]) > 0.4

        # Check collision with obstacles (boxes)
        collision = False
        for obs_pos, half_extents in self.obstacles:
            obs_min = obs_pos - half_extents - 0.1  # add small buffer
            obs_max = obs_pos + half_extents + 0.1
            if np.all(state[0:3] > obs_min) and np.all(state[0:3] < obs_max):
                collision = True
                break

        # Check timeout
        timeout = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

        return out_of_bounds or tilted or collision or timeout


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
