"""grocery controller – tightened heading control."""

# May 3, 2025

from controller import Robot
import cv2
#running yolo model trained on 300+ imgs, 200+ imgs validation with epoch = 7 
from ultralytics import YOLO
import math, random, numpy as np, collections
from math import hypot
from collections import deque
import collections

# Initialization
print("=== Initializing Grocery Shopper...")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
MAX_SPEED       = 7.0          # [rad/s]
MAX_SPEED_MS    = 0.633        # [m/s]
WHEEL_RADIUS    = MAX_SPEED_MS / MAX_SPEED
AXLE_LENGTH     = 0.4044
MOTOR_LEFT      = 10
MOTOR_RIGHT     = 11
N_PARTS         = 12

REPS            = 100
DELTA_Q         = 10
GOAL_PERCENT    = 0.05

WORLD_WIDTH     = 14.25
WORLD_HEIGHT    = 7.3
MAP_WIDTH       = 186
MAP_HEIGHT      = 360

LIDAR_ANGLE_BINS   = 667
LIDAR_SENSOR_MAX_RANGE = 5.5
LIDAR_ANGLE_RANGE  = math.radians(240)
lidar_offsets      = np.linspace(-LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_BINS)[83:-83]

#Map constants
FRAMES_BETWEEN_UPDATES = 5
PORTIONS              = 36
BUFFER                = 7

time_since_last_waypoint = 0

mode = "autonomous"

MIN_GOAL_DIST   = max(MAP_WIDTH, MAP_HEIGHT) * 1

# ---------------------------------------------------------------------------
# ROBOT DEVICES
# ---------------------------------------------------------------------------
robot     = Robot()
timestep  = int(robot.getBasicTimeStep())

keyboard  = robot.getKeyboard()
keyboard.enable(timestep)

part_names = (
    "head_2_joint","head_1_joint","torso_lift_joint","arm_1_joint","arm_2_joint",
    "arm_3_joint","arm_4_joint","arm_5_joint","arm_6_joint","arm_7_joint",
    "wheel_left_joint","wheel_right_joint",
    "gripper_left_finger_joint","gripper_right_finger_joint"
)
target_pos = (0.0,0.0,0.35,0.07,1.02,-3.16,1.27,1.32,0.0,1.41,float('inf'),float('inf'),0.045,0.045)
robot_parts = {}
for name, pos in zip(part_names, target_pos):
    dev = robot.getDevice(name)
    dev.setPosition(pos)
    dev.setVelocity(dev.getMaxVelocity()/2.0)
    robot_parts[name] = dev

left_gripper_enc  = robot.getDevice("gripper_left_finger_joint_sensor");  left_gripper_enc.enable(timestep)
right_gripper_enc = robot.getDevice("gripper_right_finger_joint_sensor"); right_gripper_enc.enable(timestep)

camera = robot.getDevice('camera'); camera.enable(timestep); camera.recognitionEnable(timestep)
gps     = robot.getDevice("gps");    gps.enable(timestep)
compass = robot.getDevice("compass"); compass.enable(timestep)
lidar   = robot.getDevice('Hokuyo URG-04LX-UG01'); lidar.enable(timestep); lidar.enablePointCloud()
display = robot.getDevice("display")

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
pose_x = pose_y = pose_theta = pose_x_last = pose_y_last = 0.0
mode            = "autonomous"
waypoints       = []
gripper_status  = "closed"

map            = np.zeros((MAP_WIDTH, MAP_HEIGHT))
collision_map  = np.zeros((MAP_WIDTH, MAP_HEIGHT))
MIN_GOAL_DIST = hypot(MAP_WIDTH, MAP_HEIGHT) * 0.071 #1/10 of the map
DRAW_COLLISION = 0
time = 0
current_frame  = 0
current_portion= PORTIONS-1
cube_x = -1
cube_y = -1

ITEMS_WORLD = [
    ( 3.49946,  7.16914, 1.07487),
    (-3.26060,  3.63000, 1.07487),
    ( 2.62029,  3.58914, 0.57487),
    ( 3.50454,  3.58914, 1.07487),
    ( 0.77720,  0.37000, 0.57487),
    (-2.66505,  0.20000, 1.07487),
    (-2.77302,  0.31000, 0.57487),
    ( 4.40093, -3.53294, 1.07487),
    ( 2.34093, -3.53294, 0.57487),
    (-1.32961, -4.05309, 0.57487),
]

# ---------------------------------------------------------------------------
# MATH HELPERS
# # ---------------------------------------------------------------------------

def convert_to_map(px, py):
    mx = MAP_WIDTH  - (MAP_WIDTH /(2*WORLD_HEIGHT)) * (py + WORLD_HEIGHT)
    my = MAP_HEIGHT - (MAP_HEIGHT/(2*WORLD_WIDTH )) * (px + WORLD_WIDTH)
    return int(max(0,min(MAP_WIDTH -1,mx))), int(max(0,min(MAP_HEIGHT-1,my)))

def convert_to_world(mx, my):
    py = (MAP_WIDTH  - mx)/(MAP_WIDTH /(2*WORLD_HEIGHT)) - WORLD_HEIGHT
    px = (MAP_HEIGHT - my)/(MAP_HEIGHT/(2*WORLD_WIDTH )) - WORLD_WIDTH
    return px, py

# ---------------------------------------------------------------------------
# RRT SUPPORT
# ---------------------------------------------------------------------------
class Node:
    def __init__(self, pt, parent=None, path_length=0):
        self.point=pt; self.parent=parent; self.path_from_parent=[]; self.path_length=path_length

def get_random_valid_vertex(_unused=None):          # ← accept a dummy arg
    """Return (x,y) that is currently free in the occupancy grid."""
    tries = 0
    while tries < 5000:
        x = np.random.randint(MAP_WIDTH)
        y = np.random.randint(MAP_HEIGHT)
        if map[x, y] < 0.20 and collision_map[x, y] == 0:
            return [x, y]
        tries += 1

    # fallback: clear inflation layer and try again
    print("⚠︎  map saturated – clearing old inflation")
    collision_map.fill(0)
    return get_random_valid_vertex()

def get_random_valid_vertex_far(sx, sy, min_dist=MIN_GOAL_DIST):
    for _ in range(5000):
        x, y = get_random_valid_vertex()
        if math.hypot(x - sx, y - sy) >= min_dist:
            return [x, y]
    # fallback to standard if none found
    return get_random_valid_vertex()

def get_nearest_vertex(nodes,pt): return min(nodes,key=lambda n:np.linalg.norm(pt-n.point))

def steer(m,p_from,p_to,delta):
    while True:
        dist=np.linalg.norm(p_to-p_from)
        if dist>delta: p_to=((p_to-p_from)*(delta/dist))+p_from
        path=np.linspace(p_from,p_to,10)
        px,py=int(round(path[1][0])),int(round(path[1][1]))
        if m[px,py]==1: p_to=np.array(get_random_valid_vertex(m))
        else: break
    for _ in range(100):
        path=np.linspace(p_from,p_to,10)
        if not any(m[round(pt[0]),round(pt[1])]==1 for pt in path): return path
        for i,pt in enumerate(path):
            if m[round(pt[0]),round(pt[1])]==1:
                p_to=(p_to-p_from)*((i)/11)+p_from
                break
    return path
    
def find_nearest_location(m, sx, sy):
    """
    Locate the nearest free cell (value 0) in m using BFS starting from (sx, sy).
    """
    visited = set()
    queue = deque()
    queue.append((sx, sy))
    visited.add((sx, sy))

    while queue:
        cur_x, cur_y = queue.popleft()

        if m[cur_x, cur_y] == 0:
            return [cur_x, cur_y]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cur_x + dx, cur_y + dy
            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return [sx, sy]

def rrtstar(m, sx, sy, ex, ey, reps, delta, gp, draw_all, draw_path, print_wp):
    # ensure start in free space
    if m[sx, sy] == 1:
        sx, sy = find_nearest_location(m, sx, sy)

    nodes = [Node(np.array([sx, sy]))]
    wpt   = []
    for _ in range(reps):
        new_pt = np.array([ex, ey]) if random.random() < gp else np.array(get_random_valid_vertex(m))
        parent = get_nearest_vertex(nodes, new_pt)
        path   = steer(m, parent.point, new_pt, delta)
        new_node = Node(path[-1], parent, parent.path_length + np.linalg.norm(path[-1] - parent.point))
        new_node.path_from_parent = path
        nodes.append(new_node)
        if np.linalg.norm(new_node.point - [ex, ey]) < 1e-5:
            break

    # trace back
    #node = nodes[-1]
    node = get_nearest_vertex(nodes, [ex, ey])
    while node:
        if draw_path:
            display.setColor(0x60FF40)
            for pt in node.path_from_parent:
                display.drawPixel(round(pt[0]), round(pt[1]))
            display.drawPixel(round(node.point[0]), round(node.point[1]))
        wpt.insert(0, (int(node.point[0]), int(node.point[1])))
        node = node.parent

    if print_wp:
        print("Waypoints:", wpt)
    np.save("path.npy", wpt)
    return wpt

def path_blocked(px, py, qx, qy, occ_grid, thresh=0.99):
    """
    occ_grid : reference to the main 'map' array (not 'collision_map').
    thresh   : occupancy probability above which a pixel is treated as solid.
    """
    dx, dy = qx - px, qy - py
    steps  = int(max(abs(dx), abs(dy)))
    if steps == 0:                       # zero‑length segment
        return occ_grid[px, py] >= thresh

    for i in range(steps + 1):
        x = int(round(px + dx * i / steps))
        y = int(round(py + dy * i / steps))
        if occ_grid[x, y] >= thresh:     # only “white” pixels block
            return True
    return False

def openGrip():
        robot.getDevice("gripper_right_finger_joint").setPosition(0.04)
        robot.getDevice("gripper_left_finger_joint").setPosition(0.04)

def closeGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.0)


from ikpy.chain import Chain
vrb = True
my_chain = Chain.from_urdf_file("robot_urdf.urdf", base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])

for link_id in range(len(my_chain.links)):

        # This is the actual link object
    link = my_chain.links[link_id]
    # I've disabled "torso_lift_joint" manually as it can cause
    # the TIAGO to become unstable.
    if link.name not in part_names or  link.name =="torso_lift_joint":
        print("Disabling {}".format(link.name))
        my_chain.active_links_mask[link_id] = False


motors = []
for link in my_chain.links:
    if link.name in part_names and link.name != "torso_lift_joint":
        motor = robot.getDevice(link.name)
        # Make sure to account for any motors that
        # require a different maximum velocity!
        if link.name == "torso_lift_joint":
            motor.setVelocity(0.07)
        else:
            motor.setVelocity(1)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        motors.append(motor)

def calculateIk(offset_target, orient=True, orientation_mode='Y', target_orientation=[0,0,1]):
    '''
    Parameters
    ----------
    offset_target : list
        A vector specifying the target position of the end effector
    orient : bool, optional
        Whether or not to orient, default True
    orientation_mode : str, optional
        Either "X", "Y", or "Z", default "Y"
    target_orientation : list, optional
        The target orientation vector, default [0,0,1]
    Returns
    -------
    list
        The calculated joint angles from inverse kinematics
    '''
    # Get the number of links in the chain
    num_links = len(my_chain.links)
    # Create initial position array with the correct size
    initial_position = [0] * num_links
    # Map each motor to its corresponding link position
    motor_idx = 0
    for i in range(num_links):
        link_name = my_chain.links[i].name
        if link_name in part_names and link_name != "torso_lift_joint":
            if motor_idx < len(motors):
                initial_position[i] = motors[motor_idx].getPositionSensor().getValue()
                motor_idx += 1
    # Calculate IK
    ikResults = my_chain.inverse_kinematics(
        offset_target, 
        initial_position=initial_position,
        target_orientation=target_orientation, 
        orientation_mode=orientation_mode
    )
    # Validate result
    position = my_chain.forward_kinematics(ikResults)
    squared_distance = math.sqrt(
            (position[0, 3] - offset_target[0])**2 + 
            (position[1, 3] - offset_target[1])**2 + 
            (position[2, 3] - offset_target[2])**2
        )
    print(f"IK calculated with error - {squared_distance}")
    return ikResults

def checkArmAtPosition(ikResults, cutoff=0.00005):
    '''Checks if arm at position, given ikResults'''
    
    # Get the initial position of the motors
    initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]
    print(f'arm pos {initial_position}')
    # Calculate the arm
    arm_error = 0
    for item in range(14):
        arm_error += (initial_position[item] - ikResults[item])**2
    arm_error = math.sqrt(arm_error)
    print(f'arm error {arm_error}')
    if arm_error < cutoff:
        if vrb:
            print("Arm at position.")
        return True
    return False

def moveArmToTarget(ikResults):
    '''Moves arm given ikResults'''
    # Set the robot motors
    for res in range(len(ikResults)):
        if my_chain.links[res].name in part_names:
            # This code was used to wait for the trunk, but now unnecessary.
            # if abs(initial_position[2]-ikResults[2]) < 0.1 or res == 2:
            robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
            if vrb:
                print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))

# --
# Init CV using ML 
#detect cubes using YOLOv8 
detect_cubes = YOLO('best_18.pt')
speed = 0 
travel_to_cube = False
detect_cubes.conf = 0.5
detect_cubes.iou = 0.4
# -- 

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
path_len=0
cube_waypoint_color = 0xFFEF00
last_waypoint = 0

while robot.step(timestep)!=-1:
    time += 1
    time_since_last_waypoint += 1
    if time % 20 == 0:
        print("mode:", mode, "time_since_last_waypoint:", time_since_last_waypoint)
    key=keyboard.getKey()
    if key==ord('A'): mode="autonomous";    waypoints.clear(); print("Switched to AUTONOMOUS mode")


    pose_x,pose_y=gps.getValues()[0],gps.getValues()[1]
    map_x,map_y=convert_to_map(pose_x,pose_y)
    n=compass.getValues(); pose_theta=math.pi-math.atan2(n[0],n[1])
    display.setColor(0x0088FF); display.drawPixel(map_x,map_y)

    # print(f'position {pose_x}, {pose_y}')
    # print(f'world {map_x}, {map_y}')

    vL=vR=MAX_SPEED/2
    current_frame+=1
    if current_frame>=FRAMES_BETWEEN_UPDATES:
        current_frame=0; current_portion=(current_portion+1)%PORTIONS
        if current_portion==0:
            if DRAW_COLLISION==1:
                display.setColor(0x008F00)
                for x in range(MAP_WIDTH):
                    for y in range(MAP_HEIGHT):
                        if collision_map[x,y]==1: display.drawPixel(x,y)
        start=current_portion*math.ceil(MAP_HEIGHT/PORTIONS)
        end=min(MAP_HEIGHT,start+math.ceil(MAP_HEIGHT/PORTIONS))
        for x in range(MAP_WIDTH):
            for y in range(start,end):
                if map[x,y]==1:
                    for dx in range(-BUFFER,BUFFER+1):
                        for dy in range(-BUFFER,BUFFER+1):
                            xi,yi=x+dx,y+dy
                            if 0<=xi<MAP_WIDTH and 0<=yi<MAP_HEIGHT:
                                collision_map[xi,yi]=1
                else:
                    display.setColor(0x000000); display.drawPixel(x,y)
        display.setColor(0x60FF40)
        for wp in waypoints:
            display.drawPixel(wp[0],wp[1])

    # ----
    # CV
    #set timestep mod to diff numbers to get more or less freq obj detection 
    if mode == "autonomous" and (time % 50 == 0): 
        print("looking for cubes")
        photo = camera.getImageArray()
        #opencv stuff
        pixels = np.array(photo, dtype=np.uint8).reshape(( camera.getHeight(),  camera.getWidth(), 3))

        converted_colors = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR) 

        results = detect_cubes(converted_colors)
        #x1, y1, x2, y2, confidence, class
        blobs = np.array(results[0].boxes.data.cpu())
        labels = detect_cubes.names

        
        for blob in blobs:
            x1, y1, x2, y2, conf, cls = blob
            center_x = (x2 + x1)/2
            center_y = (y2 + y1)/2
            label = labels[int(cls)]
            dist_from_center = math.hypot(map_x - center_x, map_y - center_y)
            print(f'{label} found with confidence ({conf}) at screen coordinates ({center_x}), ({center_y}))')
            
            if dist_from_center <=200 and conf >= 0.4:
                #set conf lower to detect cubes more frequently 
                #go towards there 
                print("SPOTTED CUBE AHEAD FOR SURE")

                #find nearest item in the array
                best_dist = 9999
                for i in ITEMS_WORLD:
                    world_x, world_y = i[0], i[1]
                    x, y = convert_to_map(world_x, world_y)
                    dist = math.hypot(map_x - x, map_y - y)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_x, best_y = x, y
                        best_cube = i
                print("The cube,", best_cube, "is at map coords", best_x, best_y)
                goto_x, goto_y = find_nearest_location(collision_map, best_x, best_y)
                print("But that is inside a wall, so actually go to", goto_x, goto_y)
                    
                mode = "travel to cube" 
                waypoints = rrtstar(collision_map, map_x, map_y,
                                    goto_x, goto_y, REPS, DELTA_Q, GOAL_PERCENT,
                                1, 1, 1)
                vL=vR=0
                print(waypoints)
                display.setColor(cube_waypoint_color); display.drawPixel(best_x, best_y)
                
                break
    # ----

    if mode == "autonomous":
        #need a path? -------------------------------------------------
        if not waypoints:
            gx, gy = get_random_valid_vertex_far(map_x, map_y)
            waypoints = rrtstar(collision_map, map_x, map_y,
                                gx, gy, REPS, DELTA_Q, GOAL_PERCENT,
                                0, 1, 1)
            path_len = len(waypoints)

    #test current leg for collision ------------------------------
    if waypoints:
        tx, ty = waypoints[0]
        #if path_blocked(map_x, map_y, tx, ty, map) or 

                #go backwards if it's been a while since we've progressed
        if time_since_last_waypoint > 450:
            vL = -max_v
            vR = -max_v

        if time_since_last_waypoint > 500: 
            print("⟳ path blocked – replanning")
            time_since_last_waypoint = 0
            #gx, gy = get_random_valid_vertex_far(map_x, map_y)          # keep same final goal
            wpx, wpy = waypoints[-1] #final goal
            waypoints = rrtstar(collision_map, map_x, map_y,
                                wpx, wpy, REPS, DELTA_Q, GOAL_PERCENT,
                                0, 1, 1)
            path_len = len(waypoints)
        tx, ty = waypoints[0]

    # If it's just been sitting there staring at a wall for a while, clearly something messed up, so just go back into autonomous
    if mode != "autonomous" and time_since_last_waypoint > 800:
        time_since_last_waypoint = 0
        mode = "autonomous"

    #drive toward first waypoint --------------------------------
    if waypoints:
        tx, ty = waypoints[0]
        dx, dy    = tx - map_x, ty - map_y
        dist_pix  = math.hypot(dx, dy)
        target_theta = (math.atan2(dy, dx) + 3*math.pi/2) % (2*math.pi)
        err = ((target_theta - pose_theta + math.pi) % (2*math.pi)) - math.pi

        wp_idx = path_len - len(waypoints) + 1
        
        # if current_frame == 0:
        #     print(f"[{wp_idx:02d}/{path_len}] pose=({map_x:3d},{map_y:3d}) θ={pose_theta:+.2f} | "
        #         f"wp=({tx},{ty}) θ={target_theta:+.2f}, dist={dist_pix:5.1f}px, err={err:+.2f}")

        close_px  = 10
        max_v     = MAX_SPEED * 0.6
        heading_tol = 0.08
        Kp        = 8.0

        if abs(err) > heading_tol:
            v_cmd = 0.0
            omega = -Kp * math.copysign(1.0, err)
            if abs(err) < 0.15:
                omega *= 0.4
        else:
            v_cmd = max_v
            omega = 0.0

        vL = v_cmd - omega * AXLE_LENGTH / 2
        vR = v_cmd + omega * AXLE_LENGTH / 2

        # waypoint reached?
        if dist_pix < close_px:
            print(f"✔ reached waypoint {wp_idx} ({tx},{ty})")
            time_since_last_waypoint = 0

            if len(waypoints) == 0:
                if travel_to_cube:
                    #stop robot 
                    travel_to_cube = False
            else:
                waypoints.pop(0)
            vL = vR = 0.0

    #if the robot reached the last waypoint during travel to cube mode, then switch to grab cube mode
    elif mode == "travel to cube":
        mode = "grab cube"
        vL = 0
        vR = 0


    # clamp wheel speeds
    ml = robot_parts["wheel_left_joint"].getMaxVelocity()
    mr = robot_parts["wheel_right_joint"].getMaxVelocity()
    scale = max(abs(vL)/ml, abs(vR)/mr, 1.0)
    vL /= scale; vR /= scale

    if mode == "grab cube":
        vL = 0
        vR = 0
        
        #rotate
        if speed<0.001:
            tx, ty    = convert_to_map(best_cube[0], best_cube[1])
            dx, dy    = tx - map_x, ty - map_y
            dist_pix  = math.hypot(dx, dy)
            target_theta = (math.atan2(dy, dx) + 3*math.pi/2) % (2*math.pi)
            err = ((target_theta - pose_theta + math.pi) % (2*math.pi)) - math.pi
            
            max_v     = MAX_SPEED * 0.6
            heading_tol = 0.08
            Kp        = 8.0

            if abs(err) > heading_tol:
                omega = -Kp * math.copysign(1.0, err)
                if abs(err) < 0.15:
                    omega *= 0.4
            else:
                v_cmd = max_v
                omega = 0.0
            vL = -omega * AXLE_LENGTH / 2
            vR =  omega * AXLE_LENGTH / 2
            if abs(err) < 0.10:
                #ANNA
                move_here = calculateIk(best_cube)
                #INVERSE KINEMATICS 
                if not checkArmAtPosition(move_here, cutoff=0.00005):
                    moveArmToTarget(move_here)
                




    # -----------------------------------------------------------------------
    # LIDAR MAPPING (unchanged)
    # -----------------------------------------------------------------------
    speed=math.hypot(pose_x-pose_x_last,pose_y-pose_y_last)
    if speed>0.005:
        readings=lidar.getRangeImage()[83:-83]
        for i,r in enumerate(readings):
            if r<LIDAR_SENSOR_MAX_RANGE*0.5:
                alpha=lidar_offsets[i]+pose_theta
                ox=-math.cos(alpha)*r; oy=math.sin(alpha)*r
                mx,my=convert_to_map(ox+pose_x,oy+pose_y)
                map[mx,my]=min(map[mx,my]+0.035,1.0)
                if map[mx,my]>=0.2:
                    for dx in range(-BUFFER,BUFFER+1):
                        for dy in range(-BUFFER,BUFFER+1):
                            xi,yi=mx+dx,my+dy
                            if 0<=xi<MAP_WIDTH and 0<=yi<MAP_HEIGHT:
                                collision_map[xi,yi]=1
                color=0xFFFFFF if map[mx,my]>=1.0 else int(map[mx,my]*255)
                display.setColor(color); display.drawPixel(mx,my)

    # -----------------------------------------------------------------------
    # DRIVE + GRIPPER
    # -----------------------------------------------------------------------
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)

    if gripper_status=="open":
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005: gripper_status="closed"
    else:
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044: gripper_status="open"

    pose_x_last,pose_y_last = pose_x,pose_y
