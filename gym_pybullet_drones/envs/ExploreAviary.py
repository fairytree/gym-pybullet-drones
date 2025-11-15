import numpy as np

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

        self.bounds = np.array([[-2, 2],        # X min/max
                                [-2, 2],        # Y min/max
                                [0.0, 2.0]])    # Z min/max
        self.visited = set()
        self.grid_size = 0.2  # discretization step in meters

        self.EPISODE_LEN_SEC = 12

        self.obstacles = [(np.array([[[1, 0, .1]]]), 0.3),
        (np.array([[[0, 1, .1]]]), 0.3),
        (np.array([[[-1, 0, .1]]]), 0.3),
        (np.array([[[0, -1, .1]]]), 0.3)]  # Define a pseudo  obstacle (position, radius)

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
        reward = 0

        # reward for visiting new cells
        if pos not in self.visited:
            reward += 1.0
            self.visited.add(pos)

        # small penalty for leaving bounds
        if np.any(state[0:3] < self.bounds[:,0]) or np.any(state[0:3] > self.bounds[:,1]):
            reward -= 0.01

        # penalty for collisions
        for obs_pos, obs_radius in self.obstacles:
            dist = np.linalg.norm(state[0:3] - obs_pos)
            if dist < obs_radius + 0.2:  # safety buffer (m)
                reward -= 0.1

        return reward

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
        out_of_bounds = (x < self.bounds[0,0]) or (x > self.bounds[0,1]) \
                        or (y < self.bounds[1,0]) or (y > self.bounds[1,1]) \
                        or (z < self.bounds[2,0]) or (z > self.bounds[2,1])

        # Check if drone is tilted too much
        tilted = abs(state[7]) > 0.4 or abs(state[8]) > 0.4

        # Check collision with obstacles
        collision = any(np.linalg.norm(state[0:3] - obs_pos) < obs_radius + 0.2
                        for obs_pos, obs_radius in self.obstacles)

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
