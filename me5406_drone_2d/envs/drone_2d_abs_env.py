import copy
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pygame


# CONSTANTS

# Physical properties of drone
DRONE_WIDTH = 0.2 # 0.2 m = 20 cm
r = DRONE_WIDTH/2 # Distance of rotor to centre of mass. Needed for physics later.
DRONE_MASS = 0.25 # In kg. (All values given in SI units)
M_OF_INERTIA = 1e-3*2 # Moment of inertia
f_g = DRONE_MASS * 9.81 # Drone weight in Newton

# Parameters for computation
TIMESTEP_LENGTH = 1/100 # This is exactly 0.01 seconds. Physics timestep length.
TIMESTEPS_PER_ACTION = 5  # One action every 5 timesteps. That's 20 actions per second
EP_LENGTH = 10 # Truncate episode after 10s
MAX_STEPS_PER_EPISODE = EP_LENGTH/(TIMESTEP_LENGTH*TIMESTEPS_PER_ACTION)

# Definition of action- and observation space
VIEWPORT = 2 # Field of 2x2 meters
MIN_SPEED = -2 # in both x and y direction
MAX_SPEED = 2
MIN_ANGLE = -np.pi # Roll angle towards right
MAX_ANGLE = np.pi # Roll angle towards left
MIN_ANG_VEL = -6.283 # Angular velocity 
MAX_ANG_VEL = 6.283 # Angular velocity 
MIN_THRUST = 0 # For both of the two rotors
MAX_THRUST = 1.5 # For each of the two rotors 

# Define the goal
GOAL_MIN = -0.1
GOAL_MAX = 0.1

# Parameters for resetting the environment (in reset() at start of every episode)
MIN_POS_R = -0.6 # Min starting position in x and y direction
MAX_POS_R = 0.6 # Max starting position in x and y direction
MIN_SPEED_R = -1.2 # In both x and y direction at episode start
MAX_SPEED_R = 1.2 # In both x and y direction at episode start 
MIN_ANGLE_R = -1.2 # Min roll angle towards right at start 
MAX_ANGLE_R = 1.2 # Max roll angle towards left at start 
MIN_ANG_VEL_R = -3.142 # Min angular velocity at start 
MAX_ANG_VEL_R = 3.142 # Max angular velocity at start 
MIN_THRUST_R = MIN_THRUST # at start for both of the two rotors
MAX_THRUST_R = MAX_THRUST # at start for each of the two rotors

# Revised Reward Structure (see report for details):
C2 = 0.0159
R2_MIN = -0.142
C3 = 0.0450
R3_MIN = -0.142
C4 = 0.0225
R4_MIN = -0.142

# For Rendering
WINDOW_SIZE = 500
PX_PER_METER = WINDOW_SIZE/VIEWPORT
GREY = (200, 200, 200)
BACKGROUND_COL = (255, 255, 255)
DRONE_COL = (0,0,100)
ROTOR_COL = (110,110,160)
SCALE = 1 # For SCALE = 1, the drone is rendered in its actual size.
    # Scetch drone for rendering:
points = np.zeros([3,10])
points[0:2,0] = [4, -2] # top right body point  	- Drone Body
points[0:2,1] = [-4, -2] # top left body point      - Drone Body
points[0:2,2] = [-4, 2] # bottom left body point    - Drone Body
points[0:2,3] = [4, 2] # bottom right body point    - Drone Body
points[0:2,4] = [-10, 0] # Main line left point     - Drone Main Line
points[0:2,5] = [10, 0] # Main line right point     - Drone Main Line
points[0:2,6] = [-13, -2] # Rotor 1 left point      - Rotor 1
points[0:2,7] = [-7, -2] # Rotor 1 right point      - Rotor 1
points[0:2,8] = [7, -2] # Rotor 2 left point        - Rotor 2
points[0:2,9] = [13, -2] # Rotor 2 right point      - Rotor 2
points = points * WINDOW_SIZE/200 # Scale properly
points = points * SCALE # For SCALE = 1, the drone is rendered in its actual size.
points[2,:] = np.ones([1,10]) # Auxiliary row of ones

class Drone2dAbsEnv(gym.Env):
    """Custom Environment that follows gym interface
    Methods: __init__(), reset(), step(), physics_engine()
    Attributes: self.action_space, self.observation_space,  self.current_state, self.render_mode
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100} 
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        # Define action and observation space
        # They must be gymnasium.spaces objects
    
        # Define action_space: The 2D drone can toggle thrust of 
        #   each of the two rotors continuously between 0 and 1 Newton.
        self.action_space = spaces.Box(low=MIN_THRUST, high=MAX_THRUST, shape=(1,2), dtype=np.float32)

        # Define observation_space: The 2D drone knows its exact position (continuous x1, x2) 
        #    and its speed (continuous v1, v2). All Toghether: 2x2
        self.observation_space = spaces.Box(low=np.array([-VIEWPORT/2, # Left Boundary
                                                        -VIEWPORT/2, # Bottom Boundary
                                                        MIN_SPEED, # Min Speed in x-direction
                                                        MIN_SPEED, # Min Speed in y-direction
                                                        MIN_ANGLE, # Min roll angle
                                                        MIN_ANG_VEL, # Min angular velocity
                                                        MIN_THRUST, # Min thrust for rotor 1
                                                        MIN_THRUST]),  # Min thrust for rotor 2
                                            high=np.array([VIEWPORT/2, # Right Boundary
                                                        VIEWPORT/2, # Top Boundary
                                                        MAX_SPEED,  # Max Speed in x-direction
                                                        MAX_SPEED, # Max Speed in y-direction
                                                        MAX_ANGLE, # Max roll angle
                                                        MAX_ANG_VEL, # Max angular velocity
                                                        MAX_THRUST, # Max thrust for rotor 1
                                                        MAX_THRUST]), # Max thrust for rotor 2
                                                        dtype=np.float32)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # For rendering
        self.window = None
        self.clock = None

    def reset(self):
        self.steps = 0
        self.current_state = self.observation_space.sample()
        # Randomize Agent and environment with some constraints to make it fair
        # self.current_state = spaces.Box(low=np.array([MIN_POS_R, # in x
        #                                             MIN_POS_R, # in y
        #                                             MIN_SPEED_R, # in x
        #                                             MIN_SPEED_R, # in y
        #                                             MIN_ANGLE_R, 
        #                                             MIN_ANG_VEL_R,
        #                                             MIN_THRUST_R, # Rotor 1
        #                                             MIN_THRUST_R]), # Rotor 2
        #                             high=np.array([MAX_POS_R, # in x
        #                                             MAX_POS_R, # in y
        #                                             MAX_SPEED_R, # in x
        #                                             MAX_SPEED_R, # in y
        #                                             MAX_ANGLE_R,
        #                                             MAX_ANG_VEL_R,
        #                                             MAX_THRUST_R, # Rotor 1
        #                                             MAX_THRUST_R]), # Rotor 2
        #                                             dtype=np.float32).sample()
        
        observation =  self.current_state
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
    #def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        # Take the action
        self.current_state[6] = action[0][0] # Set Thrust on Rotor 1
        self.current_state[7] = action[0][1] # Set Thrust on Rotor 1
        
        # Perform physics engine
        self.perform_physics_engine(TIMESTEPS_PER_ACTION, TIMESTEP_LENGTH)
        
        # Keep states within boundaries
        self.clip_state(2, MIN_SPEED, MAX_SPEED) # Speed x
        self.clip_state(3, MIN_SPEED, MAX_SPEED) # Speed y
        self.clip_state(5, MIN_ANG_VEL, MAX_ANG_VEL) # Angular velocity
    
        # Return Observation, Reward and check if terminated
        # Check if Goal reached
        if max(self.current_state[0:6]) < GOAL_MAX and min(self.current_state[0:6])>GOAL_MIN: # x Position at goal
            # Goal reached: x, y, x_dot, y_dot, theta, theta_dot < 0.1
            observation = self.current_state # Environment fully observable. Observation equals state.
            reward = 1 # Slight positive revard as extra incentive
            terminated = True
            #print("\n\n\n ----------------------Terminated---------------------- \n\n\n ")
        elif max(self.current_state[0:2]) > VIEWPORT/2 or min(self.current_state[0:2]) < -VIEWPORT/2:
            # Agent left viewport
            # Return an observation but make sure it's contained
            self.clip_state(0, -VIEWPORT/2, VIEWPORT/2) # Pos x
            self.clip_state(1, -VIEWPORT/2, VIEWPORT/2) # Pos y

            observation = self.current_state
            reward = -400 # We really don't want that
            terminated = True
        else:
            # Agent within viewport but goal not reached
            observation = self.current_state
            reward = self.get_default_reward()
            terminated = False
        
        # Episodes are truncated after 10s = 1000 timesteps = 200 decisions
        if self.steps >= MAX_STEPS_PER_EPISODE:
            truncated = True
        else:
            truncated = False

        # Get info
        info = {}

        
        self.steps +=1
        return observation, reward, terminated, truncated, info

    def get_default_reward(self):
        # Applicable when Agent within viewport but goal not reached
        r1 = -np.sqrt(self.current_state[0]**2+self.current_state[1]**2) # = -abs(x^2 + y^2)
        
        if max(self.current_state[0:2])<GOAL_MAX and min(self.current_state[0:2])>GOAL_MIN:
            # Agent is at origin. Give stabilizing rewards
            r2 = -np.sqrt(self.current_state[2]**2+self.current_state[3]**2) * C2
            r3 = -abs(self.current_state[4])* C3
            r4 = -abs(self.current_state[5])* C4
        else:
            # Agent not at origin. Do not give stabilizing rewards
            r2 = R2_MIN
            r3 = R3_MIN
            r4 = R4_MIN

        return r1 + r2 + r3 + r4

    
    def clip_state(self, idx, min, max):
        if self.current_state[idx] > max:
            self.current_state[idx] = max
        elif self.current_state[idx] < min:
            self.current_state[idx] = min
        else: 
            pass

    def perform_physics_engine(self, timesteps, dt):
        # Alters self.current_state for the duration of several timesteps.
        # No return value
        #x = self.current_state[0]
        #y = self.current_state[1]
        #x_dot = self.current_state[2]
        #y_dot = self.current_state[3]
        theta = float(self.current_state[4])
        theta_dot = float(self.current_state[5])
        f1 = float(self.current_state[6])
        f2 = float(self.current_state[7])
        f = float(f1+f2)

        p = np.array([float(self.current_state[0]), float(self.current_state[1])]) # p = [x, y]
        v = np.array([float(self.current_state[2]), float(self.current_state[3])]) # v = [x_dot, y_dot]

        # Perform pyhsics simulation
        # For detailed explanation of (1) to (5) please see report
        for _ in range(timesteps): 
             # (1)
            F_t = np.array([-np.sin(theta)*f, np.cos(theta)*f - f_g])
            a = 1/DRONE_MASS * F_t 
            tau = (f2-f1)*r
            # (2)
            p = p + v*dt 
            # (3)
            v = v + a*dt 
            # (4)
            #print(f"       {theta}      (theta before iteration with theta_dot{theta_dot}")
            theta = theta + theta_dot*dt 
            #print(f"       {theta}      (theta before iteration)")
            theta = (theta+np.pi)%(2*np.pi) - np.pi
            # (5)
            theta_ddot = 1/M_OF_INERTIA * tau
            theta_dot = theta_dot + theta_ddot*dt

            # Now update the environment
            self.current_state[0] = p[0]
            self.current_state[1] = p[1]
            self.current_state[2] = v[0]
            self.current_state[3] = v[1]
            self.current_state[4] = theta
            self.current_state[5] = theta_dot
            # Clipping will be done after this method has been called

            if self.render_mode == "human":
                self._render_frame()


        
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        canvas.fill(BACKGROUND_COL)

        # Get drone points
        pts = copy.copy(points)

        # (3) Get Transform Matrix
        theta_ = - self.current_state[4] # Coordinate system for imaging is mirrored
        c, s = np.cos(theta_), np.sin(theta_)
        T = np.zeros([3,3])
        T[0:2,0:2] = np.array(((c, -s), (s, c))) # Rotation of drone
        T[0][2] = WINDOW_SIZE/2 + self.current_state[0]*PX_PER_METER # x shift to window centre for x=0, and then add current x pos
        T[1][2] = WINDOW_SIZE/2 - self.current_state[1]*PX_PER_METER # y shift to window centre for y=0, and then add current y pos

        # (4) Apply transformation to every point
        for i in range(10):
            pts[:,i] = T.dot(pts[:,i])

        pts = pts[0:2,:] # Crop out first two lines. Those are the 2D points.

        # Here we draw stuff on canvas
        pygame.draw.line(
                canvas,
                GREY,
                (0, WINDOW_SIZE/2),
                (WINDOW_SIZE, WINDOW_SIZE/2),
                width=2,
            )
        pygame.draw.line(
                canvas,
                GREY,
                (WINDOW_SIZE/2, 0),
                (WINDOW_SIZE/2, WINDOW_SIZE),
                width=2,
            )

        # Body
        pygame.draw.polygon(canvas, DRONE_COL, (pts[:,0], pts[:,1], pts[:,2], pts[:,3]))

        # Main axis
        pygame.draw.line(canvas,DRONE_COL,pts[:,4], pts[:,5], width=3*SCALE)

        # Rotor 1
        pygame.draw.line(canvas,ROTOR_COL,pts[:,6], pts[:,7], width=3*SCALE)

        # Rotor 2
        pygame.draw.line(canvas,ROTOR_COL,pts[:,8], pts[:,9], width=3*SCALE)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



