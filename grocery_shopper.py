"""grocery controller."""

# May 1, 2025

from controller import Robot
import math
import random
import numpy as np
import collections

# Initialization
print("=== Initializing Grocery Shopper...")

# Constants
MAX_SPEED = 7.0        # [rad/s]
MAX_SPEED_MS = 0.633   # [m/s]
WHEEL_RADIUS = MAX_SPEED_MS / MAX_SPEED  # [m]
AXLE_LENGTH = 0.4044   # [m]
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

# RRT parameters
REPS = 100
DELTA_Q = 10
GOAL_PERCENT = 0.05

# World & map dimensions
WORLD_WIDTH = 14.25
WORLD_HEIGHT = 7.3
MAP_WIDTH = 186
MAP_HEIGHT = 360

# LiDAR
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5  # [m]
LIDAR_ANGLE_RANGE = math.radians(240)
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:-83]  # remove chassis‐blocked bins

# Mapping update frequency
FRAMES_BETWEEN_UPDATES = 5
PORTIONS = 36
BUFFER = 7

# Create robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Keyboard for mode switching/testing
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Set up motors & position‐controlled joints
part_names = (
    "head_2_joint","head_1_joint","torso_lift_joint","arm_1_joint","arm_2_joint",
    "arm_3_joint","arm_4_joint","arm_5_joint","arm_6_joint","arm_7_joint",
    "wheel_left_joint","wheel_right_joint","gripper_left_finger_joint","gripper_right_finger_joint"
)
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, float('inf'), float('inf'), 0.045, 0.045)
robot_parts = {}
for name, pos in zip(part_names, target_pos):
    dev = robot.getDevice(name)
    dev.setPosition(pos)
    dev.setVelocity(dev.getMaxVelocity() / 2.0)
    robot_parts[name] = dev

# Gripper encoders
left_gripper_enc = robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc = robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# GPS & compass
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Display for map visualization
display = robot.getDevice("display")

# Odometry & pose
pose_x = pose_y = pose_theta = 0.0
pose_x_last = pose_y_last = 0.0

# Mode and waypoints
#mode = "mapping"
mode = "autonomous"
waypoints = []
gripper_status = "closed"

# Map arrays
map = np.zeros((MAP_WIDTH, MAP_HEIGHT))
collision_map = np.zeros((MAP_WIDTH, MAP_HEIGHT))
DRAW_COLLISION = 0

# Frame counters
current_frame = 0
current_portion = PORTIONS - 1

# Helper: world→map coords



def convert_to_map(px, py):
    # horizontal pixel index (x‑axis) represents WORLD‑Y
    mx = MAP_WIDTH  - (MAP_WIDTH  /(2*WORLD_HEIGHT)) * (py + WORLD_HEIGHT)
    # vertical   pixel index (y‑axis) represents WORLD‑X
    my = MAP_HEIGHT - (MAP_HEIGHT /(2*WORLD_WIDTH )) * (px + WORLD_WIDTH)
    return int(max(0, min(MAP_WIDTH -1, mx))), \
           int(max(0, min(MAP_HEIGHT-1, my)))

# ---------- pixel → metres  (exact algebraic inverse) ----------
def convert_to_world(mx, my):
    py = (MAP_WIDTH  - mx) / (MAP_WIDTH  /(2*WORLD_HEIGHT)) - WORLD_HEIGHT
    px = (MAP_HEIGHT - my) / (MAP_HEIGHT /(2*WORLD_WIDTH )) - WORLD_WIDTH
    return px, py

# RRT support structures
class Node:
    def __init__(self, pt, parent=None, path_length=0):
        self.point = pt
        self.parent = parent
        self.path_from_parent = []
        self.path_length = path_length

def get_random_valid_vertex(m):
    while True:
        x = np.random.randint(MAP_WIDTH)
        y = np.random.randint(MAP_HEIGHT)
        if m[x, y] == 0:
            return [x, y]

def get_nearest_vertex(nodes, pt):
    return min(nodes, key=lambda n: np.linalg.norm(pt - n.point))

def steer(m, p_from, p_to, delta):
    while True:
        dist = np.linalg.norm(p_to - p_from)
        if dist > delta:
            p_to = ((p_to - p_from) * (delta/dist)) + p_from
        path = np.linspace(p_from, p_to, 10)
        px = int(round(path[1][0]))
        py = int(round(path[1][1]))
        if m[px, py] == 1:
            p_to = np.array(get_random_valid_vertex(m))
        else:
            break
    for _ in range(100):
        path = np.linspace(p_from, p_to, 10)
        collision = any(m[round(pt[0]), round(pt[1])] == 1 for pt in path)
        if not collision:
            return path
        # shorten if collision
        for i, pt in enumerate(path):
            if m[round(pt[0]), round(pt[1])] == 1:
                p_to = (p_to - p_from) * ((i)/11) + p_from
                break
    return path

def rrtstar(m, sx, sy, ex, ey, reps, delta, gp, draw_all, draw_path, print_wp):
    nodes = [Node(np.array([sx, sy]))]
    wpt = []
    for i in range(reps):
        new_pt = np.array([ex, ey]) if random.random()<gp else np.array(get_random_valid_vertex(m))
        parent = get_nearest_vertex(nodes, new_pt)
        path = steer(m, parent.point, new_pt, delta)
        new_node = Node(path[-1], parent, parent.path_length + np.linalg.norm(path[-1]-parent.point))
        new_node.path_from_parent = path
        nodes.append(new_node)
        if np.linalg.norm(new_node.point - [ex, ey]) < 1e-5:
            break
    # backtrack
    node = nodes[-1]
    while node:
        if draw_path:
            display.setColor(0x60FF40)
            for pt in node.path_from_parent:
                display.drawPixel(round(pt[0]), round(pt[1]))
            display.drawPixel(round(node.point[0]), round(node.point[1]))

        # >>> NEW: keep pixel indices, no convert_to_world <<<
        wpt.insert(0, (int(node.point[0]), int(node.point[1])))
        node = node.parent
    if print_wp:
        print("Waypoints:", wpt)
    np.save("path.npy", wpt)
    return wpt

# Main control loop

path_len = 0
nav_state = "turn"

while robot.step(timestep) != -1:
    # Read pose
    pose_x, pose_y = gps.getValues()[0], gps.getValues()[1]
    map_x, map_y = convert_to_map(pose_x, pose_y)
    n = compass.getValues()
    pose_theta = math.pi - math.atan2(n[0], n[1])

    # Draw robot position
    display.setColor(int(0x0088FF))
    display.drawPixel(map_x, map_y)

    # Mode switching
    key = keyboard.getKey()
    if key == ord('M'):
        mode = "mapping"
        waypoints.clear()
        print("Switched to MAPPING mode")
    elif key == ord('A'):
        mode = "autonomous"
        waypoints.clear()
        print("Switched to AUTONOMOUS mode")

    # Mapping mode: build collision map & random RRT goals
    if mode == "mapping":
        #Move forward like an idiot (for testing)
        vL = MAX_SPEED/2
        vR = MAX_SPEED/2

        current_frame += 1
        if current_frame >= FRAMES_BETWEEN_UPDATES:
            current_frame = 0
            current_portion = (current_portion + 1) % PORTIONS
            # inflate collision map once per full cycle
            if current_portion == 0:
                if DRAW_COLLISION == 1:
                    display.setColor(0x008F00)
                    for x in range(MAP_WIDTH):
                        for y in range(MAP_HEIGHT):
                            if collision_map[x, y] == 1:
                                display.drawPixel(x, y)
                            
                # new random RRT goal once per full cycle
                gx, gy = get_random_valid_vertex(collision_map)
                waypoints = rrtstar(collision_map, map_x, map_y, gx, gy,
                                    REPS, DELTA_Q, GOAL_PERCENT,
                                    draw_all=0, draw_path=1, print_wp=1)
                
            # update a slice of the map and the collision_map
            start = current_portion * math.ceil(MAP_HEIGHT/PORTIONS)
            end = min(MAP_HEIGHT, start + math.ceil(MAP_HEIGHT/PORTIONS))
            for x in range(MAP_WIDTH):
                for y in range(start, end):
                    if map[x, y] == 1:
                        for dx in range(-BUFFER, BUFFER+1):
                            for dy in range(-BUFFER, BUFFER+1):
                                xi, yi = x+dx, y+dy
                                if 0<=xi<MAP_WIDTH and 0<=yi<MAP_HEIGHT:
                                    collision_map[xi, yi] = 1
                    else:
                        # clear map pixel if not obstacle
                        display.setColor(0x000000)
                        display.drawPixel(x, y)


    # Autonomous mode: follow waypoints
    elif mode == "autonomous":

        # 1) (re)generate an RRT path if needed ----------------------
        if not waypoints:
            gx, gy = get_random_valid_vertex(collision_map)
            waypoints = rrtstar(collision_map, map_x, map_y, gx, gy,
                                REPS, DELTA_Q, GOAL_PERCENT,
                                draw_all=0, draw_path=1, print_wp=1)
            path_len = len(waypoints)        # remember how many we got

        # 2) simple angular‑error path follower ----------------------
        tx, ty      = waypoints[0]           # target waypoint (pixels)
        px, py      = map_x, map_y           # robot pose    (pixels)
        dx, dy      = tx - px, ty - py
        dist_pixels = math.hypot(dx, dy)

        target_theta = math.atan2(dy, dx)
        err = (pose_theta - target_theta + math.pi) % (2*math.pi) - math.pi

        # ---- status print with waypoint index ----------------------
        wp_idx = path_len - len(waypoints) + 1   # 1‑based counter
        print(f"[{wp_idx:02d}/{path_len}] "
              f"pose=({px:3d},{py:3d}) θ={pose_theta:+.2f} | "
              f"wp=({tx},{ty}) dist={dist_pixels:5.1f}px err={err:+.2f}")
        # ------------------------------------------------------------

        fov          = math.pi/8          # ±22.5°
        close_px     = 10                  # 4 px ≈ 10 cm
        max_v        = MAX_SPEED/3

        # reached? ---------------------------------------------------
        if dist_pixels < close_px:
            print(f"✔ reached waypoint {wp_idx} ({tx},{ty})")
            waypoints.pop(0)
            vL = vR = 0.0
            continue                      # skip to next time‑step

        # heading control -------------------------------------------
        if abs(err) <= fov:               # straight
            vL = vR = max_v
        elif err > 0:                     # turn right in place
            vL = -max_v/2;  vR =  max_v/2
        else:                             # turn left in place
            vL =  max_v/2;  vR = -max_v/2

        # 3) clamp to wheel limits ----------------------------------
        ml = robot_parts["wheel_left_joint"].getMaxVelocity()
        mr = robot_parts["wheel_right_joint"].getMaxVelocity()
        vL = max(-ml, min(vL, ml))
        vR = max(-mr, min(vR, mr))

    # LIDAR mapping update (both modes)
    speed = math.hypot(pose_x - pose_x_last, pose_y - pose_y_last)
    if speed > 0.005:
        readings = lidar.getRangeImage()[83:-83]
        for i, r in enumerate(readings):
            if r > LIDAR_SENSOR_MAX_RANGE * 0.6:        # ignore far/fuzzy hits
                continue
            alpha = lidar_offsets[i] + pose_theta
    
            # object coordinates in world space -------------------------
            ox = -math.cos(alpha) * r
            oy =  math.sin(alpha) * r
            mx, my = convert_to_map(ox + pose_x, oy + pose_y)
    
            # update probabilistic map ---------------------------------
            map[mx, my] = min(map[mx, my] + 0.03, 1.0)
    
            # -------- NEW  : immediate collision‑map "inflation" -------
            if map[mx, my] >= 0.2:                     # occupancy threshold
                for dx in range(-BUFFER, BUFFER + 1):
                    for dy in range(-BUFFER, BUFFER + 1):
                        xi, yi = mx + dx, my + dy
                        if 0 <= xi < MAP_WIDTH and 0 <= yi < MAP_HEIGHT:
                            collision_map[xi, yi] = 1
            # -----------------------------------------------------------
    
            # visualisation (blue→white with occupancy) -----------------
            color = 0xFFFFFF if map[mx, my] >= 1.0 else int(map[mx, my] * 255)
            display.setColor(color)
            display.drawPixel(mx, my)

    # Apply wheel velocities
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)

    # Gripper control (unchanged)
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

    # update for next frame
    pose_x_last, pose_y_last = pose_x, pose_y
