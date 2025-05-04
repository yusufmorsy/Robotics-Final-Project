"""grocery controller – tightened heading control."""

# May 3, 2025

from controller import Robot
import math, random, numpy as np, collections
from math import hypot
from collections import deque
import collections
from time import sleep 

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

FRAMES_BETWEEN_UPDATES = 5
PORTIONS              = 36
BUFFER                = 7

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
mode            = "shopping"
waypoints       = []
gripper_status  = "closed"

map            = np.zeros((MAP_WIDTH, MAP_HEIGHT))
collision_map  = np.zeros((MAP_WIDTH, MAP_HEIGHT))
DRAW_COLLISION = 0
current_frame  = 0
current_portion= PORTIONS-1

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

MIN_GOAL_DIST = hypot(MAP_WIDTH, MAP_HEIGHT) * 0.7  

def get_random_valid_vertex_far(sx, sy, min_dist=MIN_GOAL_DIST):
    best = None
    best_d = 0.0
    for _ in range(5000):
        x, y = get_random_valid_vertex()
        d = hypot(x - sx, y - sy)
        if d >= min_dist:
            return [x, y]       # we hit “far”!
        if d > best_d:
            best_d = d
            best   = [x, y]
    # no perfect hit—use the farthest point we saw
    print(f"⚠ no vertex ≥{min_dist:.1f}px; using farthest at {best_d:.1f}px")
    return best

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
    node = nodes[-1]
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

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
path_len=0

waypoints = []  # Initialize waypoints list
shopping_waypoints = []  # For storing shopping path waypoints
current_shopping_goal = 0  # Index of current shopping waypoint
next_shopping_goal = 1  # Index of next shopping waypoint
shopping_completed = False  # Flag to check shopping completion
shopping_started = False  # Flag to indicate if shopping routine has started

while robot.step(timestep)!=-1:
    pose_x,pose_y=gps.getValues()[0],gps.getValues()[1]
    map_x,map_y=convert_to_map(pose_x,pose_y)
    n=compass.getValues()
    pose_theta=math.pi-math.atan2(n[0],n[1])
    display.setColor(0x0088FF)
    display.drawPixel(map_x,map_y)

    key=keyboard.getKey()
    if key==ord('M'): 
        mode="mapping"
        waypoints.clear()
        print("Switched to MAPPING mode")
    if key==ord('A'): 
        mode="autonomous"
        waypoints.clear()
        print("Switched to AUTONOMOUS mode")
    if key==ord('S'):
        mode="shopping"
        # Generate shopping path from ITEMS_WORLD coordinates
        if not shopping_waypoints:
            # Start at current position
            start_w = [[pose_x, pose_y]]
            # Add all item positions
            for point in ITEMS_WORLD:
                start_w.append(point)
            
            # Create sequential path through all items
            shopping_waypoints = []
            for i in range(len(start_w) - 1):
                # Could use A* here, but for simplicity directly connecting points
                shopping_waypoints.append(start_w[i+1])
            
            print(f"Shopping path created with {len(shopping_waypoints)} waypoints")
            current_shopping_goal = 0
            next_shopping_goal = min(1, len(shopping_waypoints) - 1)
            shopping_completed = False
            shopping_started = False
        print("Switched to SHOPPING mode")

    # -----------------------------------------------------------------------
    # MAPPING MODE (unchanged)
    # -----------------------------------------------------------------------
    if mode=="mapping":
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

    # -----------------------------------------------------------------------
    # AUTONOMOUS MODE
    # -----------------------------------------------------------------------
    elif mode == "autonomous":
        # 1) need a path? -------------------------------------------------
        if not waypoints:
            gx, gy = get_random_valid_vertex_far(map_x, map_y)
            waypoints = rrtstar(collision_map, map_x, map_y,
                                gx, gy, REPS, DELTA_Q, GOAL_PERCENT,
                                0, 1, 1)
            path_len = len(waypoints)

        # 2) test current leg for collision ------------------------------
        if waypoints:
            tx, ty = waypoints[0]
            if path_blocked(map_x, map_y, tx, ty, map): 
                print("⟳ path blocked – replanning")
                gx, gy = get_random_valid_vertex_far(map_x, map_y)          # keep same final goal
                waypoints = rrtstar(collision_map, map_x, map_y,
                                    gx, gy, REPS, DELTA_Q, GOAL_PERCENT,
                                    0, 1, 1)
                path_len = len(waypoints)
            tx, ty = waypoints[0]

        # 3) drive toward first waypoint --------------------------------
        tx, ty = waypoints[0]
        dx, dy    = tx - map_x, ty - map_y
        dist_pix  = math.hypot(dx, dy)
        target_theta = (math.atan2(dy, dx) + 3*math.pi/2) % (2*math.pi)
        err = ((target_theta - pose_theta + math.pi) % (2*math.pi)) - math.pi

        wp_idx = path_len - len(waypoints) + 1
        print(f"[{wp_idx:02d}/{path_len}] pose=({map_x:3d},{map_y:3d}) θ={pose_theta:+.2f} | "
                f"wp=({tx},{ty}) θ={target_theta:+.2f}, dist={dist_pix:5.1f}px, err={err:+.2f}")

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

        # clamp wheel speeds
        ml = robot_parts["wheel_left_joint"].getMaxVelocity()
        mr = robot_parts["wheel_right_joint"].getMaxVelocity()
        scale = max(abs(vL)/ml, abs(vR)/mr, 1.0)
        vL /= scale; vR /= scale

        # waypoint reached?
        if dist_pix < close_px:
            print(f"✔ reached waypoint {wp_idx} ({tx},{ty})")
            waypoints.pop(0)
            vL = vR = 0.0

    # -----------------------------------------------------------------------
    # SHOPPING MODE (from first file)
    # -----------------------------------------------------------------------
    elif mode == "shopping":
        if shopping_completed:
            # Has finished shopping -> keep updating time but robot will be stopped forever
            vL = vR = 0.0
            continue
            
        if not shopping_started:
            # Initial routine when shopping mode starts
            vL = vR = -MAX_SPEED/3  # Back up slightly
            sleep(2)
            vL = vR = 0.0
            sleep(2)
            shopping_started = True
        
        # Don't continue if no waypoints
        if not shopping_waypoints or current_shopping_goal >= len(shopping_waypoints):
            vL = vR = 0.0
            continue
            
        # Current target waypoint in world coordinates
        target_x = shopping_waypoints[current_shopping_goal][0]
        target_y = shopping_waypoints[current_shopping_goal][1]
        
        # Next target waypoint for smoother transitions
        next_x = shopping_waypoints[next_shopping_goal][0] if next_shopping_goal < len(shopping_waypoints) else target_x
        next_y = shopping_waypoints[next_shopping_goal][1] if next_shopping_goal < len(shopping_waypoints) else target_y
        
        # Calculate errors with respect to current and goal position
        rho = math.sqrt(((pose_x - target_x)**2) + ((pose_y - target_y)**2))
        alpha = math.atan2(target_y - pose_y, target_x - pose_x) - pose_theta
        eta = math.atan2(next_y - pose_y, next_x - pose_x) - pose_theta
        
        # Handle angle wrapping
        if alpha < -3.1415: alpha += 6.283
        if eta < -3.1415: eta += 6.283
        
        # Feedback Controller (coefficients from trial/error)
        dX = rho  # Distance error
        dTheta = 5 * alpha  # Angular error with gain of 5
        
        # Inverse Kinematics Equations (vL and vR as a function dX and dTheta)
        vL = dX - (dTheta/2)  # Left wheel velocity in rad/s
        vR = dX + (dTheta/2)  # Right wheel velocity in rad/s
        
        # Proportional velocities/Clamp wheel speeds
        def cmp(a, b):
            return (a > b) - (a < b)
            
        if min(abs(vL), abs(vR)) == abs(vL):
            vL = cmp(vR, 0) * (vL/vR) * MAX_SPEED  # sign preservation with cmp(vR,0)
            vR = cmp(vR, 0) * MAX_SPEED
        else:
            vR = cmp(vR, 0) * (vR/vL) * MAX_SPEED
            vL = cmp(vR, 0) * MAX_SPEED
        
        # Reduce speed for smoother movement
        vL *= 0.6
        vR *= 0.6
        
        # Print status information
        print(f"Shopping: [{current_shopping_goal+1}/{len(shopping_waypoints)}] " 
              f"pos=({pose_x:.2f},{pose_y:.2f}) -> target=({target_x:.2f},{target_y:.2f}), "
              f"dist={rho:.2f}, alpha={alpha:.2f}, eta={eta:.2f}")
              
        # Waypoint reached?
        if rho <= 0.3 and abs(eta) <= 0.75:  # Less strict than original for smoother transitions
            print(f"✓ Reached shopping waypoint {current_shopping_goal+1}")
            
            # Check if this waypoint is an item to pick up
            is_item = False
            for item in ITEMS_WORLD:
                if math.sqrt(((target_x - item[0])**2) + ((target_y - item[1])**2)) < 0.1:
                    is_item = True
                    break
                    
            if is_item:
                # Stop and pick up the item
                vL = vR = 0.0
                print(f"Picking up item at ({target_x:.2f}, {target_y:.2f})")
                
                # Close gripper
                gripper_status = "open"  # This will trigger the gripper to close in the drive section
                
                # Wait a moment to simulate pickup
                sleep(2)
            
            # Move to next waypoint
            current_shopping_goal += 1
            next_shopping_goal = min(current_shopping_goal + 1, len(shopping_waypoints) - 1)
            
            # Check if we've completed all waypoints
            if current_shopping_goal >= len(shopping_waypoints):
                shopping_completed = True
                vL = vR = 0.0
                print("===> Finished Shopping")
                print("===> Boulder Dynamics' Grocery Shopper Has Exited.")

    # -----------------------------------------------------------------------
    # LIDAR MAPPING (unchanged)
    # -----------------------------------------------------------------------
    speed=math.hypot(pose_x-pose_x_last,pose_y-pose_y_last)
    if speed>0.005:
        readings=lidar.getRangeImage()[83:-83]
        for i,r in enumerate(readings):
            if r>LIDAR_SENSOR_MAX_RANGE*0.5: continue
            alpha=lidar_offsets[i]+pose_theta
            ox=-math.cos(alpha)*r; oy=math.sin(alpha)*r
            mx,my=convert_to_map(ox+pose_x,oy+pose_y)
            map[mx,my]=min(map[mx,my]+0.025,1.0)
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
