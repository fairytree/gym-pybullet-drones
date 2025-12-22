import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque
import time

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

DISTANCE_BUFFER_SIZE=15

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 incentive_options: dict=None
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        self.obstacles_info = [{"position": [0,0,100], "size": [0.1,0.1,0.1]}]
        self.target = np.array([0.0, 0.0, 0.0])
        self.bounds = np.array([[-3, 3],        # X min/max
                                [-3, 3],        # Y min/max
                                [0.0, 2.0]])    # Z min/max
        self.current_waypoint = np.array([0.0, 0.0, 0.0])
        self.incentive_options = incentive_options

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        
        if self.incentive_options.get("search",False):
            self.distance_buffer=np.full(DISTANCE_BUFFER_SIZE, np.linalg.norm(self.target))
            self.delta_diff_buff=np.full(DISTANCE_BUFFER_SIZE-1, np.linalg.norm(self.target))
            self.delta_delta_diff_buff=np.full(DISTANCE_BUFFER_SIZE-2, np.linalg.norm(self.target))

        
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.
        Overrides BaseAviary's method.

        """
        #clean up pybullet, removing everything except drone and plane
        DP_ids = set(self.DRONE_IDS)|{self.PLANE_ID}

        all_id = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        
        for id in all_id:
            if id not in DP_ids:
                p.removeBody(id)
        
        #clear obstacle_info
        self.obstacles_info = [{"position": [0,0,100], "size": [0.1,0.1,0.1]}]

        walls = [
            {"position": [1.0, 0.0, 0.25], "size": [0.5, 0.1, 0.5], "color": [0.8, 0.8, 0.8, 1.0]},
            {"position": [-0.5, 0.5, 0.25], "size": [0.1, 2.0, 0.5], "color": [1.0, 0.5, 0.5, 1.0]},
            {"position": [-2.5, 0.0, 0.25], "size": [1.5, 0.25, 0.5], "color": [0.3, 0.9, 0.3, 1.0]},
            {"position": [-1.5, -1.0, 0.125], "size": [0.25, 0.5, 0.25], "color": [0.5, 0.5, 1.0, 1.0]}
        ]

        if self.incentive_options.get("construct_obstacles", False):
            for wall in walls:
                self.create_wall(position=wall["position"], size=wall["size"], color=wall["color"])
                self.obstacles_info.append({"position": wall["position"], "size": wall["size"]})

        
        _theta=np.random.random()*2*np.pi

        self.target = np.array([2.3*np.cos(_theta), 2.3*np.sin(_theta), .1])
        duck_id = p.loadURDF("duck_vhacd.urdf",
            self.target,
            p.getQuaternionFromEuler([0, 0, 0]),
            globalScaling=2.0,
            physicsClientId=self.CLIENT
            )
        p.setCollisionFilterGroupMask(duck_id, -1, 1, 0) # make the duck not collidable

    def create_wall(self, position, size, color=[1, 1, 1, 1], mass=0):
        """Create a box-shaped wall in the environment."""
        
        half_extents = [size[0] / 2, size[1] / 2, size[2] / 2]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.CLIENT
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.CLIENT
        )

        wall_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.CLIENT
        )

        return wall_id

    ################################################################################
    # !!! This is where the action_space is defined, i.e., what the output looks like.
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################
    # !!! This function is to convert output of RL model to motors' RPM
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                target = self._scale_waypoint(target, self.bounds)
                state = self._getDroneStateVector(k)
                # target = state[0:3] + target[0:3]
                self.current_waypoint = target.copy()
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=0.1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
                current_time = time.time()
                if not hasattr(self, 'last_print_time'):
                    self.last_print_time = 0
                if current_time - self.last_print_time >= 1.0:
                    print("cur_pos:", np.round(state[0:3], 2),
                    "target:", np.round(target, 2), 
                    "Next pos:", np.round(next_pos, 2), 
                    "rpm_k:", np.round(rpm[0,:], 2))
                    self.last_print_time = current_time

            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _scale_waypoint(self, rl_output, bounds):
        """
        Scale RL network output from [-1,1] to environment bounds.
        
        rl_output: np.array of shape (3,) in [-1,1] for x,y,z
        bounds: np.array shape (3,2) [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
        
        Returns: np.array shape (3,) waypoint in real environment
        """
        return bounds[:, 0] + (rl_output + 1.0) * 0.5 * (bounds[:, 1] - bounds[:, 0])

    ################################################################################
    # !!! this is to construct the observation space
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ (Q means quaternion, R is roll, P is pitch, Y is yaw)
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])

            # Add space for incentives
            extra_obs_size = 0
            if self.incentive_options.get("nearest_unexplored_voxel", False):
                extra_obs_size += 4  # vec (3) + distance
            if self.incentive_options.get("exploration_percentage", False):
                extra_obs_size += 1
            if self.incentive_options.get("search",False):
                extra_obs_size+=3*DISTANCE_BUFFER_SIZE-3

            if extra_obs_size > 0:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo]*extra_obs_size for _ in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi]*extra_obs_size for _ in range(self.NUM_DRONES)])])

            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                # exclude the quaternion and keep (x, y, z, r, p, y, vx, vy, vz, wx, wy, wz)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])

            #### Add incentives ########################################
            # Determine total extra size based on enabled incentives
            extra_size = 0
            if self.incentive_options.get("nearest_unexplored_voxel", False):
                extra_size += 4  # vector3 + distance
            if self.incentive_options.get("exploration_percentage", False):
                extra_size += 1
            if self.incentive_options.get("search", False):
                extra_size += 3*DISTANCE_BUFFER_SIZE-3

            if extra_size > 0:
                extra_obs_arr = np.zeros((self.NUM_DRONES, extra_size), dtype=np.float32)
                for i in range(self.NUM_DRONES):
                    offset = 0
                    if self.incentive_options.get("nearest_unexplored_voxel", False):
                        dist, vec = self._nearest_unexplored_voxel()  # shape (1,3), scalar
                        extra_obs_arr[i, offset:offset + 3] = vec
                        extra_obs_arr[i, offset + 3] = dist
                        offset += 4
                    if self.incentive_options.get("exploration_percentage", False):
                        extra_obs_arr[i, offset] = self._exploration_percentage()
                        offset += 1
                    if self.incentive_options.get("search", False):
                        #store values
                        new_dist=np.linalg.norm(self.pos-self.target)
                        last_dist=self.distance_buffer[-1]
                        last_d_dist = self.delta_diff_buff[-1]

                        #clean dist and add
                        self.distance_buffer[:-1]=self.distance_buffer[1:]
                        self.distance_buffer[-1] = new_dist

                        #clean ddist and add
                        new_d_dist = last_dist-new_dist
                        self.delta_diff_buff[:-1]=self.delta_diff_buff[1:]
                        self.delta_diff_buff[-1] = new_d_dist        

                        #clean ddist and add
                        self.delta_delta_diff_buff[:-1]=self.delta_delta_diff_buff[1:]
                        self.delta_delta_diff_buff[-1] = last_d_dist-new_d_dist

                        concat=np.concatenate([self.distance_buffer, self.delta_diff_buff, self.delta_delta_diff_buff])

                        extra_obs_arr[i, offset:offset+concat.size] = concat
                        offset += concat.size
                ret = np.hstack([ret, extra_obs_arr])

            return ret

        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
            return None
