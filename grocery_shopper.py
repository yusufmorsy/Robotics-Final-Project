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
MAX_SPEED       = 7.0       # [rad/s]
MAX_SPEED_MS    = 0.633     # [m/s]
WHEEL_RADIUS    = MAX_SPEED_MS / MAX_SPEED  # [m]
AXLE_LENGTH     = 0.4044    # [m]
N_PARTS         = 12

# RRT★ parameters
REPS           = 1000
DELTA_Q        = 10
GOAL_PERCENT   = 0.05

# World & map dimensions
WORLD_WIDTH    = 14.25
WORLD_HEIGHT   = 7.3
MAP_WIDTH      = 186
MAP_HEIGHT     = 360

# LiDAR
LIDAR_BINS     = 667
LIDAR_MAX      = 5.5  # meters
LIDAR_ANGLE    = math.radians(240)
lidar_offsets  = np.linspace(-LIDAR_ANGLE/2, LIDAR_ANGLE/2, LIDAR_BINS)[83:-83]

# Map update
FRAMES_BETWEEN_UPDATES = 20
PORTIONS               = 36
BUFFER                 = 7

# -----------------------------------------------------------------------------
# Robot & Sensors Setup
# -----------------------------------------------------------------------------
robot     = Robot()
timestep  = int(robot.getBasicTimeStep())

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

part_names = (
    "head_2_joint","head_1_joint","torso_lift_joint","arm_1_joint","arm_2_joint",
    "arm_3_joint","arm_4_joint","arm_5_joint","arm_6_joint","arm_7_joint",
    "wheel_left_joint","wheel_right_joint",
    "gripper_left_finger_joint","gripper_right_finger_joint"
)
target_pos = (
    0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32,
    0.0, 1.41, float('inf'), float('inf'),
    0.045, 0.045
)
robot_parts = {}
for name, pos in zip(part_names, target_pos):
    dev = robot.getDevice(name)
    dev.setPosition(pos)
    dev.setVelocity(dev.getMaxVelocity()/2.0)
    robot_parts[name] = dev

# Gripper encoders
left_gripper_enc  = robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc = robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Camera
camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

# GPS & compass
gps     = robot.getDevice("gps")
compass = robot.getDevice("compass")
gps.enable(timestep)
compass.enable(timestep)

# LiDAR
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(timestep)
lidar.enablePointCloud()

# Display
display = robot.getDevice("display")

# -----------------------------------------------------------------------------
# State Variables
# -----------------------------------------------------------------------------
pose_x = pose_y = pose_theta = 0.0
pose_x_last = pose_y_last = 0.0

mode = "autonomous"   # hit 'M'/'A' to switch
waypoints = []
gripper_status = "closed"

map           = np.zeros((MAP_WIDTH, MAP_HEIGHT))
collision_map = np.zeros((MAP_WIDTH, MAP_HEIGHT))

current_frame   = 0
current_portion = PORTIONS - 1

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def convert_to_map(px, py):
    mx = MAP_HEIGHT - math.floor((MAP_HEIGHT/(2*WORLD_WIDTH))*(py+WORLD_HEIGHT)) - (MAP_HEIGHT-MAP_WIDTH)
    my = MAP_HEIGHT - math.floor((MAP_HEIGHT/(2*WORLD_WIDTH))*(px+WORLD_WIDTH))
    return int(max(0, min(MAP_WIDTH-1, mx))), int(max(0, min(MAP_HEIGHT-1, my)))

class Node:
    def __init__(self, pt, parent=None, path_length=0):
        self.point            = pt
        self.parent           = parent
        self.path_from_parent = []
        self.path_length      = path_length

def get_random_valid_vertex(cm):
    while True:
        x = np.random.randint(MAP_WIDTH)
        y = np.random.randint(MAP_HEIGHT)
        if cm[x, y] == 0:
            return [x, y]

def get_nearest_vertex(nodes, pt):
    return min(nodes, key=lambda n: np.linalg.norm(pt - n.point))

def steer(cm, p_from, p_to, delta):
    while True:
        dist = np.linalg.norm(p_to - p_from)
        if dist > delta:
            p_to = ((p_to - p_from)*(delta/dist)) + p_from
        path = np.linspace(p_from, p_to, 10)
        px, py = int(round(path[1][0])), int(round(path[1][1]))
        if cm[px, py] == 1:
            p_to = np.array(get_random_valid_vertex(cm))
        else:
            break
    for _ in range(100):
        path = np.linspace(p_from, p_to, 10)
        if all(cm[int(round(pt[0])), int(round(pt[1]))] == 0 for pt in path):
            return path
        for i, pt in enumerate(path):
            if cm[int(round(pt[0])), int(round(pt[1]))] == 1:
                p_to = (p_to - p_from)*(i/11) + p_from
                break
    return path

def rrtstar(cm, sx, sy, ex, ey, reps, delta, gp, draw_all, draw_path, print_wp):
    nodes = [Node(np.array([sx, sy]))]
    for _ in range(reps):
        sample = np.array([ex, ey]) if random.random()<gp else np.array(get_random_valid_vertex(cm))
        parent = get_nearest_vertex(nodes, sample)
        path   = steer(cm, parent.point, sample, delta)
        new    = Node(path[-1], parent, parent.path_length + np.linalg.norm(path[-1]-parent.point))
        new.path_from_parent = path
        nodes.append(new)
        if np.linalg.norm(new.point - [ex, ey]) < 1e-5:
            break

    # backtrack → world waypoints
    wpt = []
    node = nodes[-1]
    while node:
        if draw_path:
            display.setColor(0x60FF40)
            for pt in node.path_from_parent:
                display.drawPixel(round(pt[0]), round(pt[1]))
            display.drawPixel(round(node.point[0]), round(node.point[1]))
        # pixel → meters
        wx = (node.point[0]/MAP_WIDTH)*2*WORLD_WIDTH  - WORLD_WIDTH
        wy = (node.point[1]/MAP_HEIGHT)*2*WORLD_HEIGHT - WORLD_HEIGHT
        wpt.insert(0, (wx, wy))
        node = node.parent

    if print_wp:
        print("Waypoints:", wpt)
    np.save("path.npy", wpt)
    return wpt

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
while robot.step(timestep) != -1:
    # 1) Read pose
    pose_x, pose_y = gps.getValues()[0], gps.getValues()[1]
    map_x, map_y   = convert_to_map(pose_x, pose_y)
    c              = compass.getValues()
    pose_theta     = math.pi - math.atan2(c[0], c[1])

    # 2) Draw robot
    display.setColor(0x0088FF)
    display.drawPixel(map_x, map_y)

    # 3) Mode switch
    key = keyboard.getKey()
    if key == ord('M'):
        mode = "mapping";    waypoints.clear(); print("Switched to MAPPING")
    if key == ord('A'):
        mode = "autonomous"; waypoints.clear(); print("Switched to AUTONOMOUS")

    # -------------------------------------------------------------------------
    # MAPPING MODE
    # -------------------------------------------------------------------------
    if mode == "mapping":
        vL = vR = MAX_SPEED/2
        current_frame += 1
        if current_frame >= FRAMES_BETWEEN_UPDATES:
            current_frame = 0
            current_portion = (current_portion + 1) % PORTIONS
            if current_portion == 0:
                display.setColor(0x008F00)
                for x in range(MAP_WIDTH):
                    for y in range(MAP_HEIGHT):
                        if collision_map[x,y] == 1:
                            display.drawPixel(x,y)
                gx, gy = get_random_valid_vertex(collision_map)
                waypoints = rrtstar(
                    collision_map, map_x, map_y, gx, gy,
                    REPS, DELTA_Q, GOAL_PERCENT,
                    draw_all=0, draw_path=1, print_wp=1
                )
            start = current_portion * math.ceil(MAP_HEIGHT/PORTIONS)
            end   = min(MAP_HEIGHT, start + math.ceil(MAP_HEIGHT/PORTIONS))
            for x in range(MAP_WIDTH):
                for y in range(start,end):
                    if map[x,y] >= 0.5:
                        for dx in range(-BUFFER,BUFFER+1):
                            for dy in range(-BUFFER,BUFFER+1):
                                xi, yi = x+dx, y+dy
                                if 0<=xi<MAP_WIDTH and 0<=yi<MAP_HEIGHT:
                                    collision_map[xi,yi] = 1
                    else:
                        display.setColor(0x000000)
                        display.drawPixel(x,y)

    # -------------------------------------------------------------------------
    # AUTONOMOUS MODE (Lab 5 motion settings)
    # -------------------------------------------------------------------------
    
    elif mode == "autonomous":
        # rebuild collision_map
        collision_map[:] = 0
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if map[x,y] >= 0.5:
                    for dx in range(-BUFFER, BUFFER+1):
                        for dy in range(-BUFFER, BUFFER+1):
                            xi, yi = x+dx, y+dy
                            if 0 <= xi < MAP_WIDTH and 0 <= yi < MAP_HEIGHT:
                                collision_map[xi, yi] = 1

        # plan if empty
        if not waypoints:
            gx, gy = get_random_valid_vertex(collision_map)
            waypoints = rrtstar(
                collision_map, map_x, map_y, gx, gy,
                REPS, DELTA_Q, GOAL_PERCENT,
                draw_all=0, draw_path=1, print_wp=1
            )

        # print current target
        if waypoints:
            print(f"Tracking waypoint → x: {waypoints[0][0]:.3f}, y: {waypoints[0][1]:.3f}")

        # follower with relaxed thresholds
        tx, ty      = waypoints[0]
        dx, dy      = tx - pose_x, ty - pose_y
        dist        = math.hypot(dx, dy)
        target_theta= math.atan2(dy, dx)
        angle_error = (pose_theta - target_theta + math.pi) % (2*math.pi) - math.pi

        # **less sensitive** thresholds
        fov          = math.pi / 4     # now ±45°
        close_enough = 0.20            # now 20 cm
        max_cmd      = MAX_SPEED / 4   # slower turns & drives

        # arrival?
        if dist < close_enough:
            print(f"Reached waypoint → x: {tx:.3f}, y: {ty:.3f}")
            waypoints.pop(0)
            vL = vR = 0.0
        # drive straight if within wide FOV
        elif abs(angle_error) <= fov:
            vL = vR = max_cmd
        # turn right in place
        elif angle_error < 0:
            vL, vR = -max_cmd/2,  max_cmd/2
        # turn left in place
        else:
            vL, vR =  max_cmd/2, -max_cmd/2

        # clamp to actual motor limits
        ml = robot_parts["wheel_left_joint"].getMaxVelocity()
        mr = robot_parts["wheel_right_joint"].getMaxVelocity()
        vL = max(-ml, min(vL, ml))
        vR = max(-mr, min(vR, mr))


    # -------------------------------------------------------------------------
    # LIDAR → map update
    # -------------------------------------------------------------------------
    speed = math.hypot(pose_x - pose_x_last, pose_y - pose_y_last)
    if speed > 0.005:
        readings = lidar.getRangeImage()[83:-83]
        for i, r in enumerate(readings):
            if r > LIDAR_MAX * 0.6:
                continue
            alpha = lidar_offsets[i] + pose_theta
            ox    = -math.cos(alpha)*r
            oy    =  math.sin(alpha)*r
            mx, my= convert_to_map(ox+pose_x, oy+pose_y)
            val    = map[mx,my] + 0.03
            map[mx,my] = min(val,1.0)
            display.setColor(0xFFFFFF if map[mx,my]>=1.0 else int(map[mx,my]*255))
            display.drawPixel(mx,my)

    # -------------------------------------------------------------------------
    # Actuators & Gripper
    # -------------------------------------------------------------------------
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)

    if gripper_status == "open":
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue() <= 0.005:
            gripper_status = "closed"
    else:
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue() >= 0.044:
            gripper_status = "open"

    pose_x_last, pose_y_last = pose_x, pose_y
