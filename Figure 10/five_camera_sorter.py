#!/usr/bin/env python3
"""Five-camera stop-and-pick Gazebo Sim sorter (Harmonic / Jetty; no ROS 2).

    python five_camera_sorter.py world
    gz sim -r -v 3 five_camera_warehouse.sdf
    python five_camera_sorter.py run --count 30
    python five_camera_sorter.py run --count 10 --slow-camera 5 --slow-ms 260
    python five_camera_sorter.py selftest

Each arriving product receives ONE RGB frame from each of five Gazebo cameras.
All five LeNet-5 tasks are submitted after the batch is captured. A strict
200 ms wall-clock deadline begins together with the object's 200 ms nominal
camera-to-robot trip. On-time result: the object reaches the robot, the belt
STOPS, and the robot picks and drops the object and RETURNS HOME. Only then
are camera processing and new conveyor arrivals enabled for the next object.
Late/no-frame products continue off the belt; they are never picked.

One object at a time is deliberate. New objects are released at least 2 s
apart when running; pausing for the robot may increase the real interval.
Gazebo camera sensors still render, but the Python controller discards their
frames outside each object's inspection window.

The articulated robot is kinematically animated; gripping is not achieved
by contact forces or a joint-actuated physics solver.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import math
import queue
import random
import threading
import time
from collections import Counter, deque
from pathlib import Path

# ---------------------------- User settings -----------------------------
ARRIVAL_SECONDS = 2.00          # MINIMUM time between visible object releases (pause defers)
CAMERA_HZ = 5                   # 5 Hz Gazebo images, one accepted per object
CAMERA_PIXELS = 64               # lower rendering load; LeNet still receives 32x32
POOL_SIZE = 30                   # pre-create models: no runtime spawn services
POSE_HZ = 6.0                    # lower pose service load in VirtualBox
POSE_SERVICE_TIMEOUT_MS = 1800  # slow VMs get longer to answer
CAPTURE_TIMEOUT = 1.60          # gather all five frames AFTER Gazebo confirms position
POSE_ACK_TIMEOUT = 5.0            # service confirmation before classifying image frames
CAPTURE_SETTLE = 0.30             # wait for a fresh rendered frame after pose confirmation
DEADLINE_SECONDS = 0.200        # strict processing + voting + controller delivery deadline
START_X = -2.20
CAMERA_X = -0.95                # all five cameras observe the SAME item here
PICK_X = 0.85
APPROACH_SPEED = 2.5           # m/s into inspection station
FAST_TRAVEL_SECONDS = DEADLINE_SECONDS   # exactly 200 ms nominal camera -> robot
FAST_SPEED = (PICK_X - CAMERA_X) / FAST_TRAVEL_SECONDS  # 9.0 m/s
BELT_END_X = 2.68              # if not picked, product continues past robot and falls
BELT_Z = 0.53                 # center of each 0.15-m object

TRAY_X, TRAY_Y = 1.24, -1.43
TRAY_FLOOR_Z = 0.295
BINS = {'red': (-0.20, -2.35), 'green': (0.90, -2.35),
        'blue': (2.00, -2.35)}
COLORS = {'red': (.88, .13, .13), 'green': (.10, .76, .25),
          'blue': (.10, .35, .90)}

# Center positions of the three overhead and two lateral cameras.
# The lens is aimed at (CAMERA_X, 0, BELT_Z) in every case.
CAMERAS = [
    ('top_left',   (CAMERA_X, -.23, 1.46),
     (0, math.atan2(1.46-BELT_Z, .23), +math.pi/2)),
    ('top_center', (CAMERA_X, 0.00, 1.46), (0, math.pi / 2, 0)),
    ('top_right',  (CAMERA_X, +.23, 1.46),
     (0, math.atan2(1.46-BELT_Z, .23), -math.pi/2)),
    ('side_left',  (CAMERA_X, -.95, 0.89), (0, math.atan2(.89 - BELT_Z, .95), math.pi / 2)),
    ('side_right', (CAMERA_X, +.95, 0.89), (0, math.atan2(.89 - BELT_Z, .95), -math.pi / 2)),
]

ROBOT_BASE = (1.12, -.67, 0)
SHOULDER = (1.12, -.67, 1.00)
LINK_LENGTH = .75
HOME = (PICK_X, 0.00, .83)       # gripper at the pick location, raised
PICK = (PICK_X, 0.00, BELT_Z)   # gripper fingers close around the object
LIFT = (PICK_X, 0.00, 1.00)
ABOVE_TRAY = (TRAY_X, TRAY_Y, .87)
RELEASE = (TRAY_X, TRAY_Y, .48)

# The arm completes ~1.82 s of motion, fitting between 2-second arrivals.
ROBOT_STAGES = [
    ('descend', .17, HOME, PICK),
    ('close',   .13, PICK, PICK),
    ('lift',    .24, PICK, LIFT),
    ('swing',   .43, LIFT, ABOVE_TRAY),
    ('lower',   .18, ABOVE_TRAY, RELEASE),
    ('release', .20, RELEASE, RELEASE),
    ('retreat', .18, RELEASE, ABOVE_TRAY),
    ('return',  .29, ABOVE_TRAY, HOME),
]
TRAY_HOLD = .20
AUTO_TRANSFER_DURATION = .48
TICK_SECONDS = .020   # controller checks deadlines at 50 Hz; Gazebo poses run at POSE_HZ

ROOT = Path(__file__).resolve().parent
WORLD = ROOT / 'five_camera_warehouse.sdf'
WEIGHTS = ROOT / 'lenet5_demo.pt'


def rgba(c):
    return ' '.join(f'{v:.3f}' for v in c) + ' 1'


def visual(name, x, y, z, a, b, c, rgb, shape='box', rpy=(0, 0, 0)):
    if shape == 'box':
        geometry = f'<box><size>{a} {b} {c}</size></box>'
    elif shape == 'cylinder':
        geometry = f'<cylinder><radius>{a}</radius><length>{c}</length></cylinder>'
    elif shape == 'sphere':
        geometry = f'<sphere><radius>{a}</radius></sphere>'
    else:
        raise ValueError(shape)
    return (f'<visual name="{name}"><pose>{x} {y} {z} 0 0 0</pose>'
            f'<geometry>{geometry}</geometry><material>'
            f'<ambient>{rgba(rgb)}</ambient><diffuse>{rgba(rgb)}</diffuse>'
            '</material></visual>')


def model(name, xyz, geometry, extra='', rpy=(0, 0, 0)):
    pos = ' '.join(str(v) for v in (*xyz, *rpy))
    return (f'<model name="{name}"><static>true</static><pose>{pos}</pose>'
            f'<link name="link">{geometry}{extra}</link></model>')


def open_box(name, x, y, color, w=.74, d=.68):
    darker = tuple(v*.76 for v in color)
    pieces = visual('floor', 0, 0, .18, w, d, .08, darker)
    for j, yy in enumerate((-d/2, d/2)):
        pieces += visual(f'y_wall_{j}', 0, yy, .30, w, .04, .24, color)
    for j, xx in enumerate((-w/2, w/2)):
        pieces += visual(f'x_wall_{j}', xx, 0, .30, .04, d, .24, color)
    return model(name, (x, y, 0), pieces)


def make_world():
    things = []
    things.append(model('floor', (0, 0, -.055),
                         visual('slab', 0, 0, 0, 8.3, 6.9, .10, (.73, .76, .79))))
    belt = (visual('surface', 0, 0, .40, 5.10, .72, .08, (.19, .23, .26))
            + visual('rail_left', 0, -.39, .45, 5.10, .045, .14, (1., .57, .10))
            + visual('rail_right', 0, +.39, .45, 5.10, .045, .14, (1., .57, .10)))
    for n, xx in enumerate((-1.9, 1.9)):
        belt += visual(f'leg_{n}', xx, 0, .19, .10, .57, .38, (.45, .47, .49))
    things.append(model('conveyor', (0, 0, 0), belt))
    things.append(model('wall', (0, 3.15, 1.45),
                        visual('back_wall', 0, 0, 0, 8.1, .09, 3.0, (.81, .83, .85))))
    # Shelf scenery only, not H / B letter signs or poles.
    for rid, rx in enumerate((-1.35, 1.38)):
        rack = ''
        for xx in (-1.08, 1.08):
            for yy in (-.27, .27):
                rack += visual(f'upright_{xx}_{yy}', xx, yy, .79,
                               .06, .06, 1.58, (.15, .34, .54))
        for level, z in enumerate((.39, .87, 1.36)):
            rack += visual(f'beam_{level}', 0, 0, z, 2.20, .70, .05, (.38, .42, .46))
            for carton in range(4):
                rack += visual(f'carton_{level}_{carton}', -.78 + .52*carton,
                               0, z+.17, .36, .36, .28, (.60, .42, .28))
        things.append(model(f'warehouse_rack_{rid}', (rx, 1.95, 0), rack))

    # Overhead assembly has a shallow horizontal mounting plate (no pillars).
    things.append(model('camera_ceiling_mount', (CAMERA_X, 0, 1.57),
                        visual('plate', 0, 0, 0, .65, .75, .055, (.34, .37, .40))))
    for i, (label, (x, y, z), (roll, pitch, yaw)) in enumerate(CAMERAS, 1):
        # Housing points along the camera's local +X axis, as the sensor does.
        # A cylindrical rim and glass disk make the camera visibly recognizable.
        body = visual('housing', 0, 0, 0, .17, .115, .105, (.18, .21, .26))
        body += visual('lens_rim', .097, 0, 0, .050, 0, .040,
                       (.34, .38, .42), 'cylinder', rpy=(0, math.pi/2, 0))
        body += visual('lens_glass', .122, 0, 0, .034, 0, .008,
                       (.02, .08, .12), 'cylinder', rpy=(0, math.pi/2, 0))
        # The sensor optical origin is outside the housing (not inside it).
        sensor = f'''<sensor name="rgb" type="camera">
          <pose>.160 0 0 0 0 0</pose>
          <always_on>true</always_on><update_rate>{CAMERA_HZ}</update_rate>
          <topic>/sorter/cam{i}</topic>
          <camera><horizontal_fov>0.84</horizontal_fov>
          <image><width>{CAMERA_PIXELS}</width><height>{CAMERA_PIXELS}</height>
          <format>R8G8B8</format></image>
          <clip><near>0.06</near><far>7.0</far></clip></camera>
        </sensor>'''
        things.append(model(f'camera_{i}_{label}', (x, y, z), body, sensor,
                            rpy=(roll, pitch, yaw)))

    things.append(open_box('near_transfer_box', TRAY_X, TRAY_Y, (.54, .58, .63)))
    for color, (bx, by) in BINS.items():
        things.append(open_box(f'destination_{color}', bx, by, COLORS[color]))
    # A floor marker below the belt's far end indicates where missed objects fall.
    things.append(model('missed_drop_marker', (2.82, 0, 0.015),
                        visual('marker', 0, 0, 0, .58, .80, .025, (.47, .49, .51))))

    # Products are created ONCE when the world loads. On a slow VirtualBox VM
    # /create requests can time out while the camera renderer is busy.
    # Keep unused objects outside the visible room and reposition only when
    # their scheduled two-second arrival occurs.
    palette = list(COLORS)
    rng = random.Random(3)
    for pid in range(1, POOL_SIZE + 1):
        name = f'product_{pid:04d}'
        clr = rng.choice(palette)
        cube = visual('product', 0, 0, 0, .15, .15, .15, COLORS[clr])
        things.append(model(name, (0, -8, -1.0), cube))

    # A large pedestal, two independently posed mechanical links, joint
    # housings and a two-finger gripper. Inverse kinematics updates the poses.
    base = (visual('pedestal', 0, 0, .35, .18, 0, .70,
                   (.27, .31, .37), 'cylinder')
            + visual('rotation_stage', 0, 0, .73, .22, 0, .14,
                     (.34, .37, .40), 'cylinder')
            + visual('bearing', 0, 0, .95, .125, 0, .14,
                     (.22, .25, .30), 'cylinder'))
    things.append(model('robot_base', ROBOT_BASE, base))
    things.append(model('robot_upper', SHOULDER,
                        visual('main', 0, 0, 0, .092, 0, LINK_LENGTH,
                               (.96, .71, .16), 'cylinder')
                        + visual('core', 0, 0, 0, .048, 0, LINK_LENGTH+.04,
                                 (.72, .53, .15), 'cylinder')))
    things.append(model('robot_lower', SHOULDER,
                        visual('main', 0, 0, 0, .070, 0, LINK_LENGTH,
                               (.95, .67, .14), 'cylinder')))
    steel = (.28, .32, .37)
    things.append(model('robot_elbow_joint', SHOULDER,
                        visual('round', 0, 0, 0, .14, 0, 0, steel, 'sphere')))
    things.append(model('robot_wrist_joint', SHOULDER,
                        visual('round', 0, 0, 0, .12, 0, 0, steel, 'sphere')))
    things.append(model('robot_palm', (HOME[0], HOME[1], HOME[2]+.14),
                        visual('palm', 0, 0, 0, .155, .15, .11, steel)))
    for name in ('finger_left', 'finger_right'):
        things.append(model(name, HOME,
                            visual('finger', 0, 0, 0, .032, .030, .135,
                                   (.16, .19, .23))))

    sdf = '''<?xml version="1.0"?>
<sdf version="1.9"><world name="warehouse">
<gravity>0 0 -9.81</gravity>
<physics name="physics" type="ignored"><max_step_size>.005</max_step_size></physics>
<plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
<plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
<render_engine>ogre2</render_engine></plugin>
<scene><ambient>.78 .78 .78 1</ambient><background>.82 .85 .90 1</background></scene>
<light type="directional" name="sun"><pose>0 0 6 0 0 0</pose>
<diffuse>1 1 1 1</diffuse><specular>.1 .1 .1 1</specular>
<direction>-.2 .3 -.85</direction></light>
''' + '\n'.join(things) + '\n</world></sdf>\n'
    WORLD.write_text(sdf)
    print('Generated', WORLD.name)
    print('3 top cameras + 2 side cameras; no H/B signs or poles.')
    print(f'{POOL_SIZE} reusable objects loaded in the world; no runtime spawn calls.')
    print('Run: gz sim -r -v 3', WORLD.name)


def smooth(u):
    u = max(0., min(1., u))
    return u*u*(3-2*u)


def interpolate(a, b, u):
    return tuple(float(a[j]*(1-u) + b[j]*u) for j in range(3))


def robot_free(robot_stage, current_job, products):
    # A waiting product ALREADY owns the robot even before animation starts.
    return robot_stage is None and current_job is None and not any(
        p['state'] == 'waiting' for p in products)


def deadline_decision(product, now, arm_available):
    """Pure, testable decision used at the pickup position.

    A valid command must be DELIVERED by the 200 ms deadline. There is no
    grace period if a camera finishes late, even if three votes already agree.
    """
    if len(product['captures']) != 5:
        return 'missed_no_camera_frame'
    if product.get('method', 'vote') != 'vote':
        result = product.get('dist_result')
        if result is None:
            return 'missed_deadline'
        if not result.get('prediction'):
            return 'missed_detection'
        if product.get('vote_sent_at') is None or product['vote_sent_at'] > product['deadline_at']:
            return 'missed_deadline'
        if not arm_available:
            return 'missed_robot_busy'
        return 'picked'
    if len(product['results']) != 5:
        return 'missed_deadline'
    if any(not r.get('prediction') for r in product['results'].values()):
        return 'missed_detection'
    if not product.get('decision') or product.get('majority_count', 0) < 3:
        return 'missed_no_majority'
    if product.get('vote_sent_at') is None or product['vote_sent_at'] > product['deadline_at']:
        return 'missed_deadline'
    if not arm_available:
        return 'missed_robot_busy'
    return 'picked'


def can_release_next(produced, count, now, next_arrival, belt_paused,
                     robot_stage, products):
    """Avoid overlapping camera frames, arm jobs and conveyor poses."""
    return (produced < count and now >= next_arrival and not belt_paused
            and robot_stage is None
            and all(p['state'] == 'sorted' and
                    len(p.get('results', {})) == len(p.get('futures', {}))
                    for p in products))


def selftest():
    now = 100.20
    job = {'captures': {i: object() for i in range(5)},
           'results': {i: {'prediction': 'red'} for i in range(5)},
           'deadline_at': 100.20, 'vote_sent_at': 100.18,
           'decision': 'red', 'majority_count': 5}
    assert deadline_decision(job, now, True) == 'picked'
    job['vote_sent_at'] = 100.21
    assert deadline_decision(job, now, True) == 'missed_deadline'
    job['vote_sent_at'] = 100.18
    job['results'].pop(4)
    assert deadline_decision(job, now, True) == 'missed_deadline'
    job['results'][4] = {'prediction': 'red'}
    job['captures'].pop(3)
    assert deadline_decision(job, now, True) == 'missed_no_camera_frame'
    job['captures'][3] = object()
    assert deadline_decision(job, now, False) == 'missed_robot_busy'
    job['majority_count'] = 2
    assert deadline_decision(job, now, True) == 'missed_no_majority'
    assert robot_free(None, None, [{'state': 'approach'}])
    assert not robot_free(None, None, [{'state': 'waiting'}])
    assert not robot_free(0, object(), [{'state': 'approach'}])
    assert can_release_next(2, 10, 200.0, 201.0, False, None, []) is False
    assert can_release_next(2, 10, 200.0, 200.0, True, None, []) is False
    assert can_release_next(2, 10, 200.0, 200.0, False, 0, []) is False
    assert can_release_next(2, 10, 200.0, 200.0, False, None,
                            [{'state': 'tray'}]) is False
    assert can_release_next(2, 10, 200.0, 200.0, False, None,
                            [{'state': 'sorted'}]) is True
    print('[SELFTEST] Deadline, voting, pause, and one-at-a-time release passed.')


def start_controller(count, slow_camera=0, slow_ms=0., worker_count=1,
                     save_captures=False, method='vote', config_path=None):
    import numpy as np
    import struct
    import zlib
    import torch
    from torch import nn
    from torch.nn import functional as F
    try:
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.boolean_pb2 import Boolean
    except ImportError:
        from gz.transport14 import Node
        from gz.msgs11.image_pb2 import Image
        from gz.msgs11.pose_v_pb2 import Pose_V
        from gz.msgs11.boolean_pb2 import Boolean

    from distributed_lenet import DistributedLeNet, verify_equivalence
    if method != 'vote' and slow_camera:
        raise ValueError('--slow-camera is only supported for --method vote')
    if config_path is None:
        config_path = ROOT / 'partition_config.json'
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already initialized in a reused interpreter
        pass
    if count < 1 or count > POOL_SIZE:
        raise ValueError(f'--count must be from 1 to {POOL_SIZE}. '
                         'Increase POOL_SIZE and regenerate the world for more.')
    random.seed(3)
    np.random.seed(3)

    class LeNet5(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
                nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
                nn.Flatten(), nn.Linear(16*5*5, 120), nn.Tanh(),
                nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 3))

        def forward(self, x):
            return self.net(x)

    names = list(COLORS)
    net = LeNet5()
    if WEIGHTS.exists():
        net.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    else:
        print('[MODEL] Training lightweight LeNet-5 demonstration weights...')
        optimizer = torch.optim.Adam(net.parameters(), lr=.003)
        palette = np.array(list(COLORS.values()), dtype=np.float32)
        for _ in range(180):
            batch = np.random.uniform(.02, .13, (48, 3, 32, 32)).astype(np.float32)
            target = np.random.randint(0, 3, 48)
            for k, cls in enumerate(target):
                width = np.random.randint(15, 25)
                xx = np.random.randint(1, 32-width)
                yy = np.random.randint(1, 32-width)
                scale = np.random.uniform(.6, 1.1)
                batch[k, :, yy:yy+width, xx:xx+width] = np.clip(
                    palette[cls, :, None, None]*scale, 0., 1.)
            logits = net(torch.from_numpy(batch))
            loss = F.cross_entropy(logits, torch.from_numpy(target).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(net.state_dict(), WEIGHTS)
        print('[MODEL] Saved', WEIGHTS.name)
    net.eval()
    distributed = DistributedLeNet(net, config_path) if method != 'vote' else None
    if distributed is not None:
        verify_equivalence(net, config_path)
        print(f'[PARTITIONS] method={method}; configuration={config_path}')
        print('[PARTITIONS] Simulated five CPU devices, one fused 32x32 input; '
              'real partitioned conv, pool and centralized FC')
    # Do not count the first cold PyTorch execution against a product's deadline.
    with torch.inference_mode():
        for _ in range(5):
            net(torch.zeros((1, 3, 32, 32)))

    node = Node()
    events = queue.Queue(maxsize=80)
    # Camera sensors keep publishing, but frames are accepted only during
    # background initialization or the active object's capture window.
    camera_accept = threading.Event()
    camera_accept.set()

    def camera_callback(idx):
        def on_frame(msg):
            if not camera_accept.is_set():
                return
            try:
                events.put_nowait((idx, time.monotonic(), msg.width, msg.height,
                                   bytes(msg.data)))
            except queue.Full:
                pass
        return on_frame

    for i in range(5):
        if not node.subscribe(Image, f'/sorter/cam{i+1}', camera_callback(i)):
            raise RuntimeError(f'Cannot subscribe to camera /sorter/cam{i+1}; start Gazebo first')

    # IMPORTANT: pose service requests never run on the controller's deadline
    # thread. A 1-second Gazebo stall must not make a 25-ms CNN task appear late.
    pose_node = Node()
    pending_poses = queue.Queue(maxsize=1)
    pose_stop = threading.Event()
    pose_stats = {'sent': 0, 'failed': 0, 'last_ok': None}
    pose_acks = queue.Queue(maxsize=400)  # (name, ack_monotonic_time, kind)

    def pose_sender():
        last_warning = 0.0
        while not pose_stop.is_set() or not pending_poses.empty():
            try:
                vec, markers = pending_poses.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                ok, resp = pose_node.request('/world/warehouse/set_pose_vector',
                                             vec, Pose_V, Boolean,
                                             POSE_SERVICE_TIMEOUT_MS)
                success = bool(ok and resp.data)
            except Exception:
                success = False
            pose_stats['sent'] += 1
            pose_stats['last_ok'] = success
            if success:
                acked_at = time.monotonic()
                for model_name, kind in markers:
                    try:
                        pose_acks.put_nowait((model_name, acked_at, kind))
                    except queue.Full:
                        pass
            if not success:
                pose_stats['failed'] += 1
                if time.monotonic() - last_warning >= 5.0:
                    print('[WARN] Gazebo pose service still slow: '
                          'reduce camera load or allocate more VM cores')
                    last_warning = time.monotonic()

    thread_pose = threading.Thread(target=pose_sender, daemon=True,
                                   name='gazebo_pose_sender')
    thread_pose.start()

    def publish_latest_pose(vec, markers):
        # Discard an obsolete pose, never queue requests behind the simulator.
        try:
            pending_poses.put_nowait((vec, markers))
        except queue.Full:
            try:
                pending_poses.get_nowait()
            except queue.Empty:
                pass
            try:
                pending_poses.put_nowait((vec, markers))
            except queue.Full:
                pass

    def frame_array(w, h, data):
        a = np.frombuffer(data, dtype=np.uint8)
        if w <= 0 or h <= 0:
            return None
        n = a.size // (w*h)
        if n not in (3, 4) or len(a) != w*h*n:
            return None
        return a.reshape(h, w, n)[:, :, :3].copy()

    def save_png(name, frame):
        # Small lossless RGB debug image, no Pillow dependency.
        directory = ROOT / 'captures'
        directory.mkdir(exist_ok=True)
        h, w = frame.shape[:2]
        raw = b''.join(b'\x00' + frame[row].tobytes() for row in range(h))
        def chunk(tag, data):
            return (struct.pack('>I', len(data)) + tag + data +
                    struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))
        content = (b'\x89PNG\r\n\x1a\n' +
                   chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                   chunk(b'IDAT', zlib.compress(raw, 4)) + chunk(b'IEND', b''))
        (directory / name).write_bytes(content)

    background = {}

    def foreground_mask(frame, blank):
        h, w, _ = frame.shape
        x0, x1, y0, y1 = w//5, w-w//5, h//5, h-h//5
        img = frame[y0:y1, x0:x1]
        bg = blank[y0:y1, x0:x1]
        diff = np.max(np.abs(img.astype(np.int16)-bg.astype(np.int16)), axis=2)
        return img, diff > 40

    def centered_object_mask(mask):
        """Keep the foreground blob nearest the optical center, not another
        product waiting upstream in the side camera's field of view.
        """
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        best_pixels = None
        best_score = float('inf')
        for y0, x0 in zip(*np.nonzero(mask)):
            if visited[y0, x0]:
                continue
            visited[y0, x0] = True
            todo = deque([(int(y0), int(x0))])
            pixels = []
            while todo:
                y, x = todo.popleft()
                pixels.append((y, x))
                for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                    if 0 <= ny < h and 0 <= nx < w and \
                            mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        todo.append((ny, nx))
            if len(pixels) < 16:
                continue
            cy = sum(v[0] for v in pixels) / len(pixels)
            cx = sum(v[1] for v in pixels) / len(pixels)
            distance = ((cy-(h-1)/2)/(h/2))**2 + ((cx-(w-1)/2)/(w/2))**2
            # Ignore objects well away from the inspection center.
            if distance > .62:
                continue
            score = distance - .0005*len(pixels)
            if score < best_score:
                best_score, best_pixels = score, pixels
        result = np.zeros((h, w), dtype=bool)
        if best_pixels:
            ys, xs = zip(*best_pixels)
            result[ys, xs] = True
        return result

    def has_object(frame, blank):
        _, mask = foreground_mask(frame, blank)
        mask = centered_object_mask(mask)
        return int(mask.sum()) >= 24

    def prepare_camera_tensor(frame, blank):
        """Color-preserving foreground crop, identical to legacy vote model."""
        img, mask = foreground_mask(frame, blank)
        mask = centered_object_mask(mask)
        if int(mask.sum()) < 24:
            return None
        ys, xs = np.where(mask)
        xmin, xmax = max(0, int(xs.min())-4), min(img.shape[1], int(xs.max())+5)
        ymin, ymax = max(0, int(ys.min())-4), min(img.shape[0], int(ys.max())+5)
        cut = img[ymin:ymax, xmin:xmax].copy()
        fg = mask[ymin:ymax, xmin:xmax]
        cut[~fg] = 18
        tensor = torch.from_numpy(cut).permute(2, 0, 1)[None].float() / 255.
        return F.interpolate(tensor, (32, 32), mode='bilinear', align_corners=False)

    def distributed_task(captures, blanks, queued_at):
        started = time.monotonic()
        result = dict(method=method, started_at=started, error='', prediction='',
                      confidence=None, data=None, preprocess_ms=None,
                      ended_at=None, queue_ms=(started-queued_at)*1000)
        try:
            tpre = time.perf_counter()
            images = [prepare_camera_tensor(captures[i][0], blanks[i]) for i in range(5)]
            if any(tensor is None for tensor in images):
                raise ValueError('At least one camera did not detect the centered object')
            # Five viewpoints are fused into ONE image, then 5 devices
            # partition the SAME LeNet inference. This is NOT five votes.
            shared = torch.stack(images, dim=0).mean(dim=0)
            result['preprocess_ms'] = (time.perf_counter()-tpre)*1000
            with torch.inference_mode():
                data = distributed.execute(shared, method)
                probabilities = torch.softmax(data['logits'], dim=1)[0]
                k = int(torch.argmax(probabilities).item())
            result.update(data=data, prediction=names[k],
                          confidence=float(probabilities[k].item()))
        except Exception as exc:
            result['error'] = str(exc)
        result['ended_at'] = time.monotonic()
        return result

    def classify_task(camera_idx, frame, blank, queued_at):
        """Runs in one of five worker threads (no Gazebo calls here)."""
        started = time.monotonic()
        cnn_ms = None
        prediction = ''
        confidence = None
        try:
            tensor = prepare_camera_tensor(frame, blank)
            if tensor is not None:
                tnet = time.perf_counter()
                with torch.inference_mode():
                    scores = torch.softmax(net(tensor), dim=1)[0]
                cnn_ms = (time.perf_counter() - tnet)*1000
                prediction = names[int(torch.argmax(scores).item())]
                confidence = float(torch.max(scores).item())
            if slow_camera == camera_idx + 1 and slow_ms > 0:
                # Deliberate experimental latency; included in total processing time.
                time.sleep(slow_ms / 1000.)
            error = ''
        except Exception as exc:
            error = str(exc)
        ended = time.monotonic()
        return {'camera': camera_idx+1, 'prediction': prediction,
                'confidence': confidence, 'cnn_ms': cnn_ms,
                'processing_ms': (ended-started)*1000,
                'queue_ms': (started-queued_at)*1000,
                'started_at': started, 'ended_at': ended, 'error': error}

    def pose(vec, name, xyz, q=(1., 0., 0., 0.)):
        p = vec.pose.add()
        p.name = name
        p.position.x, p.position.y, p.position.z = map(float, xyz)
        p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = map(float, q)

    def mechanical_link(vec, name, a, b):
        mid = interpolate(a, b, .5)
        dx, dy, dz = (b[k]-a[k] for k in range(3))
        length = math.sqrt(dx*dx+dy*dy+dz*dz)
        ux, uy, uz = dx/length, dy/length, dz/length
        q = (1.+uz, -uy, ux, 0.)
        qlen = math.sqrt(sum(v*v for v in q))
        pose(vec, name, mid, tuple(v/qlen for v in q))

    def robot_pose(vec, target, finger_open):
        wrist = (target[0], target[1], target[2]+.22)
        delta = [wrist[j]-SHOULDER[j] for j in range(3)]
        r = math.sqrt(sum(v*v for v in delta))
        r = min(2*LINK_LENGTH-.001, max(.001, r))
        direction = [v/r for v in delta]
        center = [SHOULDER[j] + .5*r*direction[j] for j in range(3)]
        normal = [-direction[0]*direction[2],
                  -direction[1]*direction[2],
                  1.-direction[2]*direction[2]]
        mag = max(.00001, math.sqrt(sum(v*v for v in normal)))
        reach = math.sqrt(max(0., LINK_LENGTH**2 - (r/2)**2))
        elbow = tuple(center[j] + reach*normal[j]/mag for j in range(3))
        mechanical_link(vec, 'robot_upper', SHOULDER, elbow)
        mechanical_link(vec, 'robot_lower', elbow, wrist)
        pose(vec, 'robot_elbow_joint', elbow)
        pose(vec, 'robot_wrist_joint', wrist)
        pose(vec, 'robot_palm', (target[0], target[1], target[2]+.145))
        sep = .145 if finger_open else .094
        pose(vec, 'finger_left', (target[0], target[1]-sep, target[2]+.013))
        pose(vec, 'finger_right', (target[0], target[1]+sep, target[2]+.013))

    def stamp(t, t0):
        return '' if t is None else f'{t-t0:.4f}'

    def num(v, digits=3):
        return '' if v is None else f'{v:.{digits}f}'

    def sort_row(p, t0):
        elapsed = None
        if p.get('robot_at') is not None and p.get('dispatch_at') is not None:
            elapsed = 1000 * (p['robot_at'] - p['dispatch_at'])
        vote_ms = None
        if p.get('vote_sent_at') is not None:
            vote_ms = 1000 * (p['vote_sent_at']-p['dispatch_at'])
        return [p['name'], p['true'], p.get('decision', ''), p.get('outcome', ''),
                len(p['captures']), len(p['results']),
                stamp(p.get('spawn_at'), t0), stamp(p.get('dispatch_at'), t0),
                stamp(p.get('robot_at'), t0), stamp(p.get('grasp_at'), t0),
                stamp(p.get('drop_at'), t0), stamp(p.get('sorted_at'), t0),
                num(elapsed), num(vote_ms),
                num(1000*(p['pose_ack_at']-p['inspection_request_at'])
                    if p.get('pose_ack_at') else None),
                stamp(p.get('pause_at'), t0), stamp(p.get('home_at'), t0),
                num(1000*(p['home_at']-p['pause_at'])
                    if p.get('home_at') is not None and p.get('pause_at') is not None
                    else None)]

    def processing_row(p, t0):
        results = p['results']
        end_to_end = lambda i: (1000 * (results[i]['ended_at'] - p['dispatch_at'])
                                 if i in results and p.get('dispatch_at') is not None else None)
        return ([p['name'], p['true'], p.get('outcome', ''),
                 p.get('decision', ''), stamp(p.get('dispatch_at'), t0),
                 DEADLINE_SECONDS*1000, len(p['captures']),
                 len(results), num(p.get('capture_spread_ms'))]
                + [num(results[i]['processing_ms']) if i in results else '' for i in range(5)]
                + [num(results[i]['cnn_ms']) if i in results else '' for i in range(5)]
                + [num(end_to_end(i)) for i in range(5)]
                + [results[i]['prediction'] if i in results and p.get('method') == 'vote' else ''
                   for i in range(5)]
                + [num(1000*(p['vote_sent_at']-p['dispatch_at']))
                   if p.get('vote_sent_at') is not None else '',
                   'yes' if p.get('vote_sent_at') is not None
                   and p['vote_sent_at'] <= p['deadline_at'] else 'no',
                   'yes' if p.get('decision') == p['true'] else 'no'])

    print('[CAMERAS] Waiting for five EMPTY-background camera images...')
    background_deadline = time.monotonic() + 35.0
    while len(background) < 5:
        if time.monotonic() > background_deadline:
            raise RuntimeError(f'Only {len(background)} of 5 cameras sent images. '
                               'Check gz topic -l and ensure Gazebo is UNPAUSED.')
        try:
            i, ts, w, h, raw = events.get(timeout=2.0)
        except queue.Empty:
            print('[CAMERAS] Waiting...')
            continue
        frame = frame_array(w, h, raw)
        if frame is not None:
            background.setdefault(i, frame)
            print(f'[CAMERAS] {len(background)}/5 backgrounds', end='\r', flush=True)
    camera_accept.clear()
    print('\n[CAMERAS] Ready. Camera processing pauses between objects.')

    # File rows are flushed continuously, even when a product misses the arm.
    with (ROOT/'camera_votes.csv').open('w', newline='') as vote_file, \
         (ROOT/'sort_log.csv').open('w', newline='') as sort_file, \
         (ROOT/'object_processing_times.csv').open('w', newline='') as timing_file, \
         (ROOT/'distributed_object_times.csv').open('w', newline='') as dist_file, \
         (ROOT/'distributed_layer_times.csv').open('w', newline='') as layers_file, \
         (ROOT/'distributed_messages.csv').open('w', newline='') as messages_file, \
         ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='lenet_camera') as workers:
        vote_csv, sort_csv, timing_csv = (csv.writer(vote_file), csv.writer(sort_file),
                                         csv.writer(timing_file))
        vote_csv.writerow(['product', 'camera', 'position', 'prediction', 'confidence',
                           'cnn_ms', 'processing_ms', 'queue_ms',
                           'completion_after_capture_ms', 'completed_before_deadline',
                           'matches_true_color', 'error'])
        sort_csv.writerow(['product', 'true_color', 'voted_color', 'result',
                           'frames_captured', 'tasks_completed', 'arrival_s', 'dispatch_s',
                           'robot_arrival_s', 'grasp_s', 'drop_s', 'finished_s',
                           'camera_to_robot_ms', 'decision_latency_ms',
                           'inspection_pose_ack_ms', 'belt_paused_s',
                           'arm_home_s', 'belt_pause_ms'])
        timing_csv.writerow(['product', 'true_color', 'result', 'voted_color',
                             'dispatch_s', 'deadline_ms', 'frames_captured',
                             'tasks_completed', 'camera_frame_spread_ms']
                            + [f'camera{i}_processing_ms' for i in range(1,6)]
                            + [f'camera{i}_cnn_ms' for i in range(1,6)]
                            + [f'camera{i}_completion_ms' for i in range(1,6)]
                            + [f'camera{i}_prediction' for i in range(1,6)]
                            + ['vote_and_send_ms', 'within_deadline', 'vote_correct'])
        dist_csv = csv.DictWriter(dist_file, fieldnames=[
            'product', 'method', 'true_color', 'prediction', 'result',
            'camera_count', 'preprocess_ms', 'device1_compute_ms', 'device2_compute_ms',
            'device3_compute_ms', 'device4_compute_ms', 'device5_compute_ms',
            'total_compute_ms', 'fc_merge_ms', 'network_bytes', 'network_messages',
            'network_sum_ms', 'network_critical_ms', 'modeled_parallel_ms',
            'measured_distributed_ms', 'worker_queue_ms', 'decision_latency_ms',
            'within_deadline', 'error'])
        dist_csv.writeheader()
        layer_csv = csv.DictWriter(layers_file, fieldnames=[
            'product', 'method', 'device', 'layer', 'output_start', 'output_end',
            'input_start', 'input_end', 'computed_elements', 'compute_ms', 'remote_bytes',
            'remote_messages', 'communication_ms'])
        layer_csv.writeheader()
        message_csv = csv.DictWriter(messages_file, fieldnames=[
            'product', 'method', 'layer', 'source_device', 'destination_device',
            'bytes', 'estimated_ms'])
        message_csv.writeheader()
        for f in (vote_file, sort_file, timing_file, dist_file, layers_file, messages_file):
            f.flush()
        print(f'[WORKERS] {worker_count} CNN worker(s) for 5 camera tasks; '
              'inference and completion latency are logged separately')

        all_products = []
        current_job = None
        robot_stage = None
        stage_started = None
        tool = HOME
        fingers_open = True
        produced = 0
        counts = Counter()
        t0 = time.monotonic()
        next_spawn = t0 + .4
        next_status = t0 + 8
        next_pose_due = 0.0
        robot_pose_initialized = False
        belt_paused = False
        home_pose_pending = False
        home_job = None

        def record_sort(p):
            # Never log a pickup as a completed sort. The home ACK may
            # arrive before the tray has completed its automatic transfer.
            if p['state'] != 'sorted':
                return
            if p.get('outcome') == 'sorted' and p.get('home_at') is None:
                return
            if not p.get('sort_logged'):
                sort_csv.writerow(sort_row(p, t0))
                sort_file.flush()
                p['sort_logged'] = True

        def record_processing(p):
            if p.get('sort_logged') and not p.get('timing_logged') and \
                    len(p['results']) == len(p['futures']):
                timing_csv.writerow(processing_row(p, t0))
                timing_file.flush()
                if method != 'vote':
                    d = p.get('dist_result') or {}
                    data = d.get('data') or {}
                    perdev = data.get('device_compute_ms', [None]*5)
                    dist_csv.writerow({
                        'product': p['name'], 'method': method,
                        'true_color': p['true'], 'prediction': d.get('prediction', ''),
                        'result': p.get('outcome', ''), 'camera_count': len(p['captures']),
                        'preprocess_ms': num(d.get('preprocess_ms')),
                        **{f'device{i+1}_compute_ms': num(perdev[i]) for i in range(5)},
                        'total_compute_ms': num(data.get('sum_compute_ms')),
                        'fc_merge_ms': num(data.get('fc_ms')),
                        'network_bytes': data.get('network_bytes', ''),
                        'network_messages': data.get('network_messages', ''),
                        'network_sum_ms': num(data.get('network_sum_ms')),
                        'network_critical_ms': num(data.get('network_critical_ms')),
                        'modeled_parallel_ms': num(data.get('parallel_estimate_ms')),
                        'measured_distributed_ms': num(data.get('measured_inference_ms')),
                        'worker_queue_ms': num(d.get('queue_ms')),
                        'decision_latency_ms': num(1000*(p['vote_sent_at']-p['dispatch_at'])
                                                    if p.get('vote_sent_at') else None),
                        'within_deadline': 'yes' if p.get('vote_sent_at') and
                             p['vote_sent_at'] <= p['deadline_at'] else 'no',
                        'error': d.get('error', '')})
                    dist_file.flush()
                p['timing_logged'] = True

        try:
            while True:
                now = time.monotonic()
                if can_release_next(produced, count, now, next_spawn,
                                    belt_paused, robot_stage, all_products):
                    produced += 1
                    name = f'product_{produced:04d}'
                    color = random.choice(names)
                    # Matching models are already present in the world.
                    all_products.append(dict(name=name, true=color, method=method,
                                             spawn_at=time.monotonic(),
                                             state='approach', captures={},
                                             futures={}, results={},
                                             xyz=(START_X, 0, BELT_Z)))
                    print(f'[ARRIVAL] {name} ({color}), next after >=2 s '
                          'and only when previous robot cycle is complete')
                    next_spawn = now + ARRIVAL_SECONDS

                # A camera capture is legal ONLY after Gazebo confirms the object
                # reached its inspection pose. The old controller used the
                # mathematical position even after a Gazebo request timed out.
                for _ in range(200):
                    try:
                        model_name, ack_at, kind = pose_acks.get_nowait()
                    except queue.Empty:
                        break
                    if model_name == 'robot_palm' and kind == 'home':
                        if home_pose_pending:
                            home_pose_pending = False
                            belt_paused = False
                            if home_job is not None:
                                home_job['home_at'] = ack_at
                                print(f'[ARM HOME] {home_job["name"]}: Gazebo '
                                      'confirmed robot return to home position')
                                record_sort(home_job)
                                home_job = None
                            print('[BELT RESUME] Robot home confirmed; next '
                                  'object/camera cycle enabled')
                        continue
                    matching = next((p for p in all_products
                                     if p['name'] == model_name), None)
                    if matching is None:
                        continue
                    if kind == 'inspection' and matching['state'] == 'capture_pending':
                        matching['state'] = 'capture'
                        matching['pose_ack_at'] = ack_at
                        matching['capture_wait_at'] = ack_at + CAPTURE_SETTLE
                        camera_accept.set()
                        print(f'[POSE OK] {model_name}: inspection pose confirmed; '
                              'waiting for fresh camera frames')
                    elif kind == 'final' and matching['state'] == 'sorted':
                        matching['final_pose_ack'] = True

                # Harvest camera images only AFTER the pose acknowledgement and
                # a render-settle interval. Exactly one frame per camera/object.
                for _ in range(110):
                    try:
                        i, ts, w, h, raw = events.get_nowait()
                    except queue.Empty:
                        break
                    candidates = [p for p in all_products
                                  if p['state'] == 'capture' and i not in p['captures']
                                  and ts >= p['capture_wait_at']]
                    if not candidates:
                        continue
                    frame = frame_array(w, h, raw)
                    if frame is None or not has_object(frame, background[i]):
                        continue
                    candidates[0]['captures'][i] = (frame, ts)
                    if save_captures:
                        save_png(f'{candidates[0]["name"]}_cam{i+1}.png', frame)

                # Collect finished tasks without blocking motion or the deadline.
                for p in all_products:
                    if method != 'vote':
                        future = p['futures'].get(0)
                        if future is not None and future.done() and not p.get('dist_recorded'):
                            p['dist_recorded'] = True
                            try:
                                d = future.result()
                            except Exception as exc:
                                d = {'prediction': '', 'error': str(exc),
                                     'ended_at': time.monotonic(), 'data': None}
                            p['dist_result'] = d
                            data = d.get('data') or {}
                            perdev = data.get('device_compute_ms', [None]*5)
                            # Compatibility with the original five-column CSV:
                            # these are DEVICE COMPUTE values, not five independent CNNs.
                            for i in range(5):
                                p['results'][i] = dict(
                                    prediction=d.get('prediction',''),
                                    confidence=d.get('confidence'),
                                    cnn_ms=perdev[i], processing_ms=perdev[i],
                                    queue_ms=d.get('queue_ms'),
                                    ended_at=d['ended_at'], error=d.get('error',''))
                            p['vote_at'] = time.monotonic()
                            if d.get('prediction'):
                                p['decision'] = d['prediction']
                                p['vote_sent_at'] = p['vote_at']  # delivery after main-thread harvest
                            for e in data.get('layer_events', []):
                                layer_csv.writerow(dict(
                                    product=p['name'], method=method,
                                    device=e['device'], layer=e['layer'],
                                    output_start=e['start_row'], output_end=e['end_row'],
                                    input_start=e['input_start'], input_end=e['input_end'],
                                    computed_elements=e['output_elements'],
                                    compute_ms=num(e['compute_ms']),
                                    remote_bytes=e['remote_bytes'],
                                    remote_messages=e['remote_messages'],
                                    communication_ms=num(e['communication_ms'])))
                            layers_file.flush()
                            for m in data.get('messages', []):
                                message_csv.writerow(dict(
                                    product=p['name'], method=method,
                                    layer=m['layer'], source_device=m['source'],
                                    destination_device=m['destination'],
                                    bytes=m['bytes'], estimated_ms=num(m['estimated_ms'])))
                            messages_file.flush()
                            print(f'[DISTRIBUTED] {p["name"]} [{method}]: '
                                  f'{d.get("prediction") or "no_detection"}; '
                                  f'sum_compute={num(data.get("sum_compute_ms"))} ms, '
                                  f'network={num(data.get("network_critical_ms"))} ms modeled, '
                                  f'delivery={1000*(p["vote_at"]-p["dispatch_at"]):.1f} ms'
                                  f'{" ERROR="+d["error"] if d.get("error") else ""}')
                        continue
                    for i, future in list(p['futures'].items()):
                        if i in p['results'] or not future.done():
                            continue
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {'camera': i+1, 'prediction': '', 'confidence': None,
                                      'cnn_ms': None, 'processing_ms': None,
                                      'queue_ms': None, 'ended_at': time.monotonic(),
                                      'error': str(exc)}
                        p['results'][i] = result
                        elapsed_ms = 1000*(result['ended_at']-p['dispatch_at'])
                        vote_csv.writerow([p['name'], i+1, CAMERAS[i][0],
                                           result['prediction'], num(result['confidence'], 4),
                                           num(result['cnn_ms']), num(result['processing_ms']),
                                           num(result['queue_ms']), num(elapsed_ms),
                                           'yes' if result['ended_at'] <= p['deadline_at'] else 'no',
                                           'yes' if result['prediction'] == p['true'] else 'no',
                                           result['error']])
                        vote_file.flush()
                        print(f'[CAM {i+1}] {p["name"]}: '
                              f'{result["prediction"] or "no_detection"}, '
                              f'{num(result["processing_ms"])} ms '
                              f'({"on-time" if result["ended_at"] <= p["deadline_at"] else "LATE"})')
                    # A majority vote requires ALL FIVE results; the outcome is
                    # not allowed to change retroactively after the deadline.
                    if p.get('dispatch_at') is not None and \
                            len(p['results']) == 5 and p.get('vote_at') is None:
                        if all(r['prediction'] for r in p['results'].values()):
                            labels = [p['results'][i]['prediction'] for i in range(5)]
                            tally = Counter(labels)
                            winner, votes = tally.most_common(1)[0]
                            p['vote_counts'] = dict(tally)
                            p['majority_count'] = votes
                            p['vote_at'] = time.monotonic()
                            if votes >= 3:
                                p['decision'] = winner
                                # Local controller dispatch (no network simulation).
                                p['vote_sent_at'] = p['vote_at']
                                print(f'[VOTE] {p["name"]}: {dict(tally)} => '
                                      f'{p["decision"]}; '
                                      f'delivery={1000*(p["vote_sent_at"]-p["dispatch_at"]):.1f} ms')
                            else:
                                print(f'[VOTE] {p["name"]}: {dict(tally)} => NO MAJORITY')

                vec = Pose_V()
                active = False
                for p in all_products:
                    state = p['state']
                    if state == 'approach':
                        # Exactly one object travels on the conveyor per cycle.
                        x = min(CAMERA_X, START_X + APPROACH_SPEED*(now-p['spawn_at']))
                        p['xyz'] = (x, 0, BELT_Z)
                        if x >= CAMERA_X:
                            p['state'] = 'capture_pending'
                            p['inspection_request_at'] = now
                            print(f'[INSPECT] {p["name"]}: requesting confirmed Gazebo '
                                  'inspection pose')
                    elif state == 'capture_pending':
                        p['xyz'] = (CAMERA_X, 0, BELT_Z)
                        if now-p['inspection_request_at'] >= POSE_ACK_TIMEOUT:
                            p['state'] = 'sorted'
                            camera_accept.clear()
                            p['outcome'] = 'sim_pose_timeout'
                            # Never let an unconfirmed product remain inside
                            # the next object's real camera image.
                            p['xyz'] = (0, -9.0-produced*.20, -1.0)
                            p['sorted_at'] = now
                            print(f'[SIM MISS] {p["name"]}: inspection pose NOT confirmed '
                                  f'within {POSE_ACK_TIMEOUT:g}s; no classification attempted')
                            record_sort(p)
                    elif state == 'capture':
                        p['xyz'] = (CAMERA_X, 0, BELT_Z)
                        if len(p['captures']) == 5 or now-p['capture_wait_at'] >= CAPTURE_TIMEOUT:
                            camera_accept.clear()   # stop processing additional camera frames
                            p['state'] = 'fast'
                            p['dispatch_at'] = time.monotonic()
                            p['deadline_at'] = p['dispatch_at'] + DEADLINE_SECONDS
                            if p['captures']:
                                image_times = [im[1] for im in p['captures'].values()]
                                p['capture_spread_ms'] = 1000 * (max(image_times)-min(image_times))
                            else:
                                p['capture_spread_ms'] = None
                            if method == 'vote':
                                for i, (frame, ts) in p['captures'].items():
                                    p['futures'][i] = workers.submit(
                                        classify_task, i, frame, background[i], p['dispatch_at'])
                            elif len(p['captures']) == 5:
                                future = workers.submit(distributed_task, dict(p['captures']),
                                                        background, p['dispatch_at'])
                                p['futures'] = {i: future for i in range(5)}
                            print(f'[CAPTURE] {p["name"]}: {len(p["captures"])}/5 frames, '
                                  f'method={method}, 200 ms deadline and fast belt')
                    elif state == 'fast':
                        elapsed = now-p['dispatch_at']
                        x = min(PICK_X, CAMERA_X + FAST_SPEED*elapsed)
                        p['xyz'] = (x, 0, BELT_Z)
                        if elapsed >= FAST_TRAVEL_SECONDS:
                            p['robot_at'] = now
                            outcome = deadline_decision(
                                p, now, robot_free(robot_stage, current_job, all_products))
                            if outcome == 'picked':
                                p['state'] = 'waiting'
                                p['outcome'] = 'picked'
                                p['pause_at'] = now
                                belt_paused = True
                                camera_accept.clear()
                                print(f'[BELT STOP] {p["name"]}: '
                                      f'{p["decision"]} ON TIME; object held at robot, '
                                      'camera processing OFF, starting pickup')
                            else:
                                p['state'] = 'missed_running'
                                p['outcome'] = outcome
                                print(f'[MISSED] {p["name"]}: {outcome}; '
                                      'the object continues to the conveyor end')
                    elif state == 'missed_running':
                        elapsed = now-p['robot_at']
                        x = min(BELT_END_X, PICK_X+FAST_SPEED*elapsed)
                        p['xyz'] = (x, 0, BELT_Z)
                        if x >= BELT_END_X:
                            p['state'] = 'falling'
                            p['fall_at'] = now
                    elif state == 'falling':
                        elapsed = now-p['fall_at']
                        z = max(.075, BELT_Z-4.905*elapsed**2)
                        p['xyz'] = (BELT_END_X+.05, 0, z)
                        if z <= .075:
                            p['state'] = 'sorted'
                            p['sorted_at'] = now
                            print(f'[FELL] {p["name"]}: reached the floor at belt end')
                            record_sort(p)
                    elif state == 'gripped':
                        p['xyz'] = tool
                    elif state == 'dropping':
                        elapsed = now-p['drop_at']
                        z = max(TRAY_FLOOR_Z, RELEASE[2]-4.905*elapsed**2)
                        p['xyz'] = (TRAY_X, TRAY_Y, z)
                    elif state == 'tray':
                        if now-p['tray_at'] >= TRAY_HOLD:
                            p['state'] = 'auto_transfer'
                            p['transfer_at'] = now
                            p['from'] = p['xyz']
                            idx = counts[p['decision']]
                            counts[p['decision']] += 1
                            bx, by = BINS[p['decision']]
                            p['to'] = (bx+.17*((idx % 3)-1),
                                       by+.17*(((idx//3) % 3)-1),
                                       TRAY_FLOOR_Z + .16*(idx//9))
                    elif state == 'auto_transfer':
                        u = (now-p['transfer_at']) / AUTO_TRANSFER_DURATION
                        p['xyz'] = interpolate(p['from'], p['to'], smooth(u))
                        if u >= 1.:
                            p['state'] = 'sorted'
                            p['sorted_at'] = now
                            p['outcome'] = 'sorted'
                            print(f'[SORTED] {p["name"]} -> {p["decision"]}')
                            record_sort(p)

                # A product is never queued beyond its pickup point. If the
                # arm is busy at that instant, it is marked missed above.
                if robot_stage is None:
                    jobs = [p for p in all_products if p['state'] == 'waiting']
                    if jobs:
                        current_job = jobs[0]
                        robot_stage = 0
                        stage_started = now
                        print(f'[ARM] Reaching for {current_job["name"]}')

                if robot_stage is not None:
                    name, duration, start, end = ROBOT_STAGES[robot_stage]
                    u = min(1., (now-stage_started)/duration)
                    tool = interpolate(start, end, smooth(u))
                    fingers_open = (name in ('descend', 'release', 'retreat', 'return'))
                    if name == 'close':
                        fingers_open = u < .5
                    if u >= 1.:
                        if name == 'close':
                            current_job['state'] = 'gripped'
                            current_job['grasp_at'] = now
                            print(f'[GRASP] {current_job["name"]}')
                        elif name == 'lower':
                            current_job['state'] = 'dropping'
                            current_job['drop_at'] = now
                            current_job['xyz'] = RELEASE
                            print(f'[DROP] {current_job["name"]} into near box')
                        elif name == 'release':
                            current_job['state'] = 'tray'
                            current_job['tray_at'] = now
                            current_job['xyz'] = (TRAY_X, TRAY_Y, TRAY_FLOOR_Z)
                        robot_stage += 1
                        stage_started = now
                        if robot_stage >= len(ROBOT_STAGES):
                            home_job = current_job
                            home_job['return_command_at'] = now
                            home_pose_pending = True
                            robot_stage = None
                            current_job = None
                            tool = HOME
                            fingers_open = True
                            print('[ARM] Return animation finished; waiting '
                                  'for Gazebo home-position confirmation')

                # Update camera/robot deadline logic at 50 Hz, but send a
                # combined Gazebo pose snapshot at most POSE_HZ times/second.
                # The pose service never blocks classification scheduling.
                for p in all_products:
                    if p['state'] == 'gripped':
                        p['xyz'] = tool
                    record_processing(p)

                if now >= next_pose_due:
                    markers = []
                    # Don't repeat a robot arm pose when the arm is idle.
                    if robot_stage is not None or home_pose_pending or not robot_pose_initialized:
                        robot_pose(vec, tool, fingers_open)
                        robot_pose_initialized = True
                        if home_pose_pending:
                            markers.append(('robot_palm', 'home'))
                    for p in all_products:
                        if p['state'] == 'sorted' and p.get('final_pose_ack'):
                            continue  # don't send already acknowledged final pose
                        pose(vec, p['name'], p['xyz'])
                        if p['state'] == 'capture_pending':
                            markers.append((p['name'], 'inspection'))
                        if p['state'] == 'sorted':
                            markers.append((p['name'], 'final'))
                    # Never re-send a room full of objects still hidden off-screen.
                    if len(vec.pose):
                        publish_latest_pose(vec, markers)
                    next_pose_due = now + 1.0 / POSE_HZ
                wait = TICK_SECONDS-(time.monotonic()-now)
                if wait > 0:
                    time.sleep(wait)
                if now >= next_status:
                    next_status = now+8
                    complete = sum(p['state'] == 'sorted' for p in all_products)
                    misses = sum(p.get('outcome', '').startswith(('missed', 'sim_')) for p in all_products)
                    print(f'[STATUS] arrivals={produced} finished={complete} '
                          f'misses={misses} belt={"STOPPED" if belt_paused else "RUNNING"}')
                if produced >= count and all(p['state'] == 'sorted' and
                         p.get('timing_logged') for p in all_products):
                    print('[DONE] All products finished and all camera timings logged.')
                    break
        except KeyboardInterrupt:
            print('\n[STOP] Saving camera timings collected so far.')
        finally:
            pose_stop.set()
            thread_pose.join(timeout=POSE_SERVICE_TIMEOUT_MS/1000 + .35)
            print(f'[POSE STATUS] sent={pose_stats["sent"]} '
                  f'failures={pose_stats["failed"]}')
            # If stopped mid-run, record results available at that time.
            for p in all_products:
                if not p.get('sort_logged'):
                    if not p.get('outcome'):
                        p['outcome'] = 'interrupted'
                    record_sort(p)
                if not p.get('timing_logged') and p.get('dispatch_at') is not None:
                    timing_csv.writerow(processing_row(p, t0))
            for f in (vote_file, sort_file, timing_file, dist_file, layers_file, messages_file):
                f.flush()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=['world', 'run', 'selftest'])
    cli.add_argument('--method', choices=('vote', 'independent', 'halo'), default='vote',
                     help='vote=original 5 separate CNNs; other modes=one distributed CNN')
    cli.add_argument('--config', default=str(ROOT/'partition_config.json'),
                     help='five-device row partition and link configuration (JSON)')
    cli.add_argument('--count', type=int, default=10,
                     help=f'number of products, 1..{POOL_SIZE} (default 10)')
    cli.add_argument('--workers', type=int, choices=(1, 2, 3, 4, 5), default=1,
                     help='CNN workers: 1 for small VMs, 5 for concurrent tasks')
    cli.add_argument('--save-captures', action='store_true',
                     help='save accepted 64x64 camera images as PNG for debugging')
    cli.add_argument('--pool-info', action='store_true',
                     help='show the number of products pre-created in the world')
    cli.add_argument('--slow-camera', type=int, choices=range(0, 6), default=0,
                     metavar='0..5', help='camera to slow artificially (0 = none)')
    cli.add_argument('--slow-ms', type=float, default=0.,
                     help='add this many milliseconds of processing to --slow-camera')
    args = cli.parse_args()
    if args.pool_info:
        print(f'POOL_SIZE={POOL_SIZE}; regenerate world after changing this constant')
        return
    if args.mode == 'world':
        make_world()
    elif args.mode == 'selftest':
        selftest()
        from distributed_lenet import verify_equivalence
        import torch
        from torch import nn
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
                    nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
                    nn.Flatten(), nn.Linear(16*5*5, 120), nn.Tanh(),
                    nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 3))
            def forward(self, x): return self.net(x)
        torch.set_num_threads(1)
        verify_equivalence(TestNet(), args.config)
        print('[SELFTEST] Both partitioned LeNet methods match full model output.')
    else:
        start_controller(args.count, args.slow_camera, args.slow_ms,
                         args.workers, args.save_captures, args.method, args.config)


if __name__ == '__main__':
    main()
