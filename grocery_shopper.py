"""grocery controller."""

# May 1, 2025

from controller import Robot
import math
import random
import collections
import numpy as np

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Keyboard for testing
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

#Size of the world (in radius)
world_width  = 14.25
world_height = 7.3

#Size of the map (1px is 0.08m)
map_width  = 186
map_height = 360

#Variables used for calculating speed
pose_x_last_frame = 0
pose_y_last_frame = 0

#RRT variables
reps = 1000
delta_q = 10
goal_percent = 0.05
goal_x = 160
goal_y = 160

#Map variables
frames_between_updates = 20 #30 frames in a second
current_update_frame = 0
portions = 36
current_portion = portions-1
multiplier = math.ceil(map_height/portions)
waypoints = []

map           = np.zeros(shape=[map_width, map_height]) #The map that appears on screen
collision_map = np.zeros(shape=[map_width, map_height]) #The actual map that the robot sees (normal map inflated by the robot's radius)

buffer = 7 #radius of the collision map filling

mode = "mapping"

#Default things
lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis
gripper_status="closed"

# ------------------------------------------------------------------
# Helper Functions
def convert_to_map(pose_x, pose_y):
    #Convert values
    x = map_height-math.floor((map_height/(2*world_width))*(pose_y+world_height))-(map_height-map_width)
    y = map_height-math.floor((map_height/(2*world_width))*(pose_x+world_width))

    #Clamp values
    x = min(x, map_width-1)
    y = min(y, map_height-1)
    x = max(x, 0)
    y = max(y, 0)
    return [x, y]

class Node:
    def __init__(self, pt, parent=None, path_length=0):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
        self.path_length = path_length

def get_random_valid_vertex(map):
    vertex = None
    while vertex is None:
        pt_x = np.random.randint(map_width)
        pt_y = np.random.randint(map_height)
        if (map[pt_x][pt_y] == 0):
            vertex = [pt_x, pt_y]
    return vertex

def get_nearest_vertex(node_list, q_point):
    nearest_node = None
    nearest_dist = 99999
    for node in node_list:
        dist = np.linalg.norm(q_point - node.point)
        if dist < nearest_dist:
            nearest_node = node
            nearest_dist = dist
    return nearest_node

def steer(map, from_point, to_point, delta_q):
    while True:
        dist = np.linalg.norm(from_point - to_point)
        if dist > delta_q:
            to_point = ((to_point - from_point) * (delta_q / dist)) + from_point
        path = np.linspace(from_point, to_point, 10)

        # roll a new point if the first section of the path is blocked
        part = path[1]
        xx = round(part[0])
        yy = round(part[1])
        if (map[xx][yy] == 1):
            to_point = get_random_valid_vertex(map)
        else:
            break

    exit = 0
    for i in range(100):
        path = np.linspace(from_point, to_point, 10)
        if exit:
            break
        for i in range(10):
            part = path[i]
            xx = round(part[0])
            yy = round(part[1])
            if (map[xx][yy] == 0):
                exit += 1
            else:
                to_point = (to_point - from_point) * ( (i + 1) / 11) + from_point
                exit = 0
                break
    return path

def draw_pixels(draw, color):
    display.setColor(color)
    for i in range(len(draw)):
        if len(draw) == 2:
            part = draw
        else:
            part = draw[i]
        display.drawPixel(round(part[0]), round(part[1]))

def rrtstar(map, start_x, start_y, end_x, end_y, reps, delta_q, goal_percent, draw_all, draw_path, print_waypoints):
    node_list = [Node(np.array([start_x, start_y]), parent=None)]
    waypoints = []

    for i in range(reps):
        # Get valid spot for a point to go
        if random.random() < goal_percent:
            new_node = Node(np.array([end_x, end_y]))
        else:
            new_node = Node(get_random_valid_vertex(map))
        parent = get_nearest_vertex(node_list, new_node.point)
        new_node.parent = parent
        new_node.path_from_parent = steer(map, parent.point, new_node.point, delta_q)
        new_node.point = new_node.path_from_parent[9]

        # Find all nearby nodes
        nearby_nodes = []
        for node in node_list:
            dist = np.linalg.norm(new_node.point - node.point)
            if dist <= delta_q:
                nearby_nodes.append(node)   

        # Find best parent for the new node
        for node in nearby_nodes:
            if np.linalg.norm(steer(map, node.point, new_node.point, delta_q)[9] - new_node.point) < 1e-5: #If there are no obstacles
                dist_to_node   = np.linalg.norm(new_node.point - node.point)
                dist_to_parent = np.linalg.norm(new_node.point - parent.point)
                if node.path_length + dist_to_node < parent.path_length + dist_to_parent:
                    parent = node
                    new_node.path_from_parent = np.linspace(parent.point, new_node.point, 10)
            
        # Assign variables
        new_node.parent = parent
        new_node.path_from_parent = np.linspace(parent.point, new_node.point, 10)
        new_node.path_length = np.linalg.norm(new_node.point - parent.point) + parent.path_length
        node_list.append(new_node)
                
        # See if the nearby nodes would rather change their parent to the new node
        for node in nearby_nodes:
            parent = node.parent
            if parent is not None: #If it's not the starting node
                if np.linalg.norm(steer(map, node.point, new_node.point, delta_q)[9] - new_node.point) < 1e-5: #If there are no obstacles
                    dist_to_new    = np.linalg.norm(node.point - new_node.point)
                    dist_to_parent = np.linalg.norm(node.point - parent.point)
                    if new_node.path_length + dist_to_new < parent.path_length + dist_to_parent:
                        node.path_from_parent = np.linspace(node.point, new_node.point, 10)
                        node.parent = new_node

        # Finalize and draw
        node_list.append(new_node)
        if draw_all:
            draw_pixels(new_node.path_from_parent, int(0xFFF000))
            draw_pixels(new_node.point, int(0xFF0000))

        # Quit if a valid path is found
        if np.linalg.norm(new_node.point - [end_x, end_y]) < 1e-5:
            break

    while new_node is not None:
        if draw_path:
            draw_pixels(new_node.path_from_parent, int(0x60FF40))
            draw_pixels(new_node.point, int(0x60FF40))
        x = new_node.point[0]
        y = new_node.point[1]
        waypoints.insert(0, (-12*(1-(x/360)), -12*(y/360)))
        new_node = new_node.parent

    if print_waypoints:
        print("Waypoints: ")
        for point in waypoints:
            print(point)

    np.save("path.npy", waypoints)
    return waypoints

# Main Loop
while robot.step(timestep) != -1:
    
    #Get pose and map position
    pose_x, pose_y = gps.getValues()[0], gps.getValues()[1]
    map_x,  map_y  = convert_to_map(pose_x, pose_y)
    n = compass.getValues()
    pose_theta = 3.1415-(math.atan2(n[0], n[1]))

    #Draw position on map in light blue
    display.setColor(int(0x0088FF))
    display.drawPixel(map_x, map_y)

    ###Debug tools
    key = keyboard.getKey()

    #Save map manually
    if key == ord('S'):
        np.save("map.npy", map)
        print("Map file saved")

    #Load map manually
    if key == ord('L'):
        map = np.load("map.npy")
        print("Map loaded")

        display.setColor(int(0x000000))
        for i in range(map_width):
            for j in range(map_height):
                display.drawPixel(i, j)

        display.setColor(int(0xFFFFFF))
        for i in range(map_width):
            for j in range(map_height):
                if map[i][j] == 1:
                    display.drawPixel(i, j)

    #Do RRT*
    if key == ord('R'):
        waypoints = rrtstar(collision_map, map_x, map_y, goal_x, goal_y, reps, delta_q, goal_percent, 1, 1, 1)

    #Draw collision map
    if key == ord('C'):
        display.setColor(0x008F00)
        for xpos in range(map_width):
            for ypos in range(map_height):
                if collision_map[xpos][ypos] == 1:
                    display.drawPixel(xpos, ypos)

    if mode == "mapping":
        #Move forward like an idiot (this needs to be changed to following the waypoints)
        vL = MAX_SPEED/2
        vR = MAX_SPEED/2

        #Update map periodically
        current_update_frame += 1

        #Once a second, update a portion of the map
        if (current_update_frame >= frames_between_updates):
            current_portion += 1
            current_portion = current_portion % portions
            current_update_frame = 0

            #Once every full cycle, draw the updated collision map and do a new RRT*
            if current_portion == 0:
                #Visually update collision map 
                display.setColor(0x008F00)
                for xpos in range(map_width):
                    for ypos in range(map_height):
                        if collision_map[xpos][ypos] == 1:
                            display.drawPixel(xpos, ypos)

                #Do RRT* to a random location
                goal = get_random_valid_vertex(collision_map)
                waypoints = rrtstar(collision_map, map_x, map_y, goal[0], goal[1], reps, delta_q, goal_percent, 0, 1, 1)

            for xpos in range(map_width):
                for ypos in range(current_portion*multiplier, (current_portion+1)*multiplier):
                    #If there's a full detection, add it to the collision map
                    if map[xpos][ypos] == 1:
                        display.setColor(int(0xFFFFFF))
                        for k in range(2*buffer):
                            for l in range(2*buffer):
                                if (xpos + k - buffer >= 0 and ypos + l - buffer >= 0 and xpos + k - buffer < map_width and ypos + l - buffer < map_height):
                                    collision_map[xpos + k - buffer][ypos + l - buffer] = 1

                    #If there's a partial detection, remove it
                    else:
                        map[xpos][ypos] = 0
                        display.setColor(0x000000)
                    display.drawPixel(xpos, ypos)


        #If the robot is moving, then get and draw LIDAR readings
        speed = math.sqrt(abs(pose_x - pose_x_last_frame)**2 + abs(pose_y - pose_y_last_frame)**2)
        speed_threshold = 0.005 #Full throttle forward is about 0.02
        if (speed > speed_threshold):
            lidar_sensor_readings = lidar.getRangeImage()
            lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
            for i, rho in enumerate(lidar_sensor_readings):
                alpha = lidar_offsets[i] #angle compared to the robot's heading
                beta = alpha + pose_theta #angle of the sensor in world view

                #Don't do anything if the object is too far away
                if rho > LIDAR_SENSOR_MAX_RANGE*(0.6): #the multiplier is there because things get fuzzy at the end of the range
                    continue

                #Object's offset from the robot
                offset_x = -math.cos(beta)*rho
                offset_y =  math.sin(beta)*rho

                #Convert to map coords
                xpos, ypos = convert_to_map(offset_x + pose_x, offset_y + pose_y)
                
                #Draw on map and set to map array
                value = map[xpos][ypos]
                map[xpos][ypos] = map[xpos][ypos] + 0.03
                #Clamp to 1, and make it white if it's maxxed
                if map[xpos][ypos] > 1: 
                    map[xpos][ypos] = 1
                    display.setColor(0xFFFFFF)
                #If not 1, then set some shade of blue
                else:
                    display.setColor(int(map[xpos][ypos] * 255))
                display.drawPixel(xpos, ypos)
                
    #Update variables used for speed calculation (needs to go at the very end of the main code loop)
    pose_x_last_frame, pose_y_last_frame = pose_x, pose_y
    
    #Default code
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
