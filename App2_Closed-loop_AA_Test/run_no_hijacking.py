#IMPORT PART
import glob
import os
import sys
import imageio
from typing import List, Optional, Tuple
from cv2 import perspectiveTransform
import matplotlib.pyplot as plt

from kornia import vec_like
try:
    sys.path.append(glob.glob('carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import time
import math
# import csv

import argparse
import cv2
import numpy as np
import queue

import Nvidia_agent

import torch
import random

import getkeys
import getpoint
#IMPORT PART END

class Node:
    def __init__(self, x: float, y: float, yaw: float, v: float) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
    
    def update(self, x: float, y: float, yaw: float, v: float) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class C:
    # PID config
    Kp = 0.3  # proportional gain

    # system config
    Ld = 2.6  # look ahead distance default 2.6
    kf = 0.1  # look forward gain
    dt = 0.1  # T step
    dist_stop = 0.7  # stop distance
    dc = 0.0

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.9  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width
    MAX_STEER = 0.30
    MAX_ACCELERATION = 5.0

def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))

HEIGHT = 200
WIDTH = 200

actor_list = []
client = None
vehicle = None
walker = None
sensor = None
world = None
image_queue = None
agent = None
savebackvout = None
saveupvout = None

def show_img(image):
    image = np.array(image.raw_data)
    image = image.reshape((HEIGHT,WIDTH,4))
    image_queue.queue.clear()
    image_queue.put_nowait(image)

def show_back_view(img):
    global savebackvout
    image = np.array(img.raw_data)
    image = image.reshape((1080, 1920, 4))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    savebackvout.write(image)
    cv2.imshow("back_view", cv2.resize(image, (240, 135)))
    cv2.waitKey(1)

def show_up_view(img):
    global saveupvout
    image = np.array(img.raw_data)
    image = image.reshape((1080, 1920, 4))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    saveupvout.write(image)
    cv2.imshow("up_view", cv2.resize(image, (240, 135)))
    cv2.waitKey(1)

def init_world():
    global actor_list, client, vehicle, world, image_queue, sensor, walker
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world = client.load_world("Town02")
    time.sleep(1)
    world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))
    time.sleep(1)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds=0.0
    world.apply_settings(settings)
    time.sleep(1)
    # add vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('model3')
    blueprint = vehicle_blueprints[0]
    point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(blueprint, point)
    time.sleep(1)
    print(vehicle.type_id)
    actor_list.append(vehicle)
    world.tick()

    spectator = world.get_spectator()
    spectator.set_transform(get_transform(vehicle.get_location(), -90))
    world.tick()
    # front view of the vehicle
    image_queue = queue.Queue(1)
    cam_bp = blueprint_library.filter("sensor.camera.rgb")[0]
    cam_bp.set_attribute("image_size_x", "{}".format(HEIGHT))
    cam_bp.set_attribute("image_size_y", "{}".format(WIDTH))
    cam_bp.set_attribute("fov","90")
    spawn_point = carla.Transform(carla.Location(x=1.5,z=2.4))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data: show_img(data))
    world.tick()
    # third view of the vehicle
    cam_bp = blueprint_library.filter("sensor.camera.rgb")[0]
    cam_bp.set_attribute("image_size_x", "{}".format(1920))
    cam_bp.set_attribute("image_size_y", "{}".format(1080))
    cam_bp.set_attribute("fov","110")
    spawn_point = carla.Transform(carla.Location(x=-5.0,z=3.4))
    back_sensor = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
    actor_list.append(back_sensor)
    back_sensor.listen(lambda data: show_back_view(data))
    world.tick()
    cam_bp = blueprint_library.filter("sensor.camera.rgb")[0]
    cam_bp.set_attribute("image_size_x", "{}".format(1920))
    cam_bp.set_attribute("image_size_y", "{}".format(1080))
    cam_bp.set_attribute("fov","110")
    spawn_point = carla.Transform(carla.Location(x=1.5,z=8.0), carla.Rotation(pitch = -90))
    up_sensor = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
    actor_list.append(up_sensor)
    up_sensor.listen(lambda data: show_up_view(data))
    world.tick()
    # add pedestrian
    '''
    blueprintsWalkers = blueprint_library.filter("walker.pedestrian.*")
    walker_bp = random.choice(blueprintsWalkers)
    rotation = point.rotation
    rotation.yaw += 90
    point = carla.Transform(carla.Location(-3.0, 160, 0.5), rotation)
    walker = world.spawn_actor(walker_bp, point)
    time.sleep(1)
    actor_list.append(walker)
    world.tick()
    print(walker.get_location(), vehicle.get_location())
    '''


def init_agent():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4", action="store_true", help="store mp4")
    args = parser.parse_args()
    args.path_folder_model = "model_RL_IAs_CARLA_Challenge"
    args.path_folder_model = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), args.path_folder_model
    )
    args.steps_image = [-10,-2,-1,0,]
    args.crop_sky = False
    args.device = torch.device("cuda")
    args.disable_cuda = False
    torch.cuda.manual_seed(random.randint(1, 10000))
    torch.backends.cudnn.enabled = True
    args.nb_action_throttle = 3
    args.nb_action_steering = 27
    args.quantile_embedding_dim = 64
    args.front_camera_height = 288
    args.front_camera_width = 288
    args.render = True
    args.num_quantile_samples = 32
    args.max_steering = 0.6
    args.max_throttle = 1.0
    # agent = agent_IAs_RL.AgentIAsRL(args)
    agent = Nvidia_agent.Nvidia_agent(args)

def init_utils():
    global savebackvout, saveupvout
    fps = 10
    size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    savebackvout = cv2.VideoWriter()
    savebackvout.open(time.strftime('back_view/%Y%m%d-%H-%M-%S.mp4', time.localtime()), fourcc, fps, size, True)
    saveupvout = cv2.VideoWriter()
    saveupvout.open(time.strftime('up_view/%Y%m%d-%H-%M-%S.mp4', time.localtime()), fourcc, fps, size, True)
    cv2.imshow("back_view", np.zeros((135, 240), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.imshow("up_view", np.zeros((135, 240), dtype=np.uint8))
    cv2.waitKey(1)

class PATH:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.ind_end = len(self.cx) - 1
        self.index_old = None

    def target_index(self, node):
        """
        search index of target point in the reference path.
        the distance between target point and current position is ld
        :param node: current information
        :return: index of target point
        """

        if self.index_old is None:
            self.calc_nearest_ind(node)

        Lf = C.kf * node.v + C.Ld

        for ind in range(self.index_old, self.ind_end + 1):
            if self.calc_distance(node, ind) > Lf:
                self.index_old = ind
                return ind, Lf

        self.index_old = self.ind_end

        return self.ind_end, Lf

    def calc_nearest_ind(self, node):
        """
        calc index of the nearest point to current position
        :param node: current information
        :return: index of nearest point
        """

        dx = [node.x - x for x in self.cx]
        dy = [node.y - y for y in self.cy]
        ind = np.argmin(np.hypot(dx, dy))
        self.index_old = ind

    def calc_distance(self, node, ind):
        return math.hypot(node.x - self.cx[ind], node.y - self.cy[ind])

def pure_pursuit(node, ref_path, index_old):
    """
    pure pursuit controller
    :param node: current information
    :param ref_path: reference path: x, y, yaw, curvature
    :param index_old: target index of last time
    :return: optimal steering angle
    """

    ind, Lf = ref_path.target_index(node)  # target point and pursuit distance
    ind = max(ind, index_old)

    tx = ref_path.cx[ind]
    ty = ref_path.cy[ind]            

    alpha = math.atan2(ty - node.y, tx - node.x) - node.yaw
    delta = math.atan2(2.0 * C.WB * math.sin(alpha), Lf)

    return delta, ind

def getPath() -> Tuple[List[float], List[float]]:
    f = open('path.txt', 'r')
    lines = f.readlines()
    path_x = []
    path_y = []
    for line in lines:
        path_x.append(float(line.split()[0]))
        path_y.append(float(line.split()[1]))
    f.close()
    return (path_x[0:-1:10], path_y[0:-1:10])

class Arrow:
    def __init__(self, x, y, theta, L, c):
        angle = np.deg2rad(30)
        d = 0.4 * L
        w = 2

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + math.pi - angle
        theta_hat_R = theta + math.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)


class Graph:
    def __init__(self, path_x: Optional[List[float]], path_y: Optional[List[float]]) -> None:
        self.record_x = []
        self.record_y = []
        self.record_yaw = []
        self.p_x = []
        self.p_y = []
        self.p_yaw = []
        self.filenames = []
        self.pedestrian = True
        if path_x != None:
            self.path_x = [-x for x in path_x]
            self.path_y = path_y
            self.pedestrian = False

    def add_graph(self, x: float, y: float, yaw: float) -> None:
        '''
        no pedestrian and hijack the vehicle according to the given trajectory
        '''
        self.record_x.append(-x)
        self.record_y.append(y)
        self.record_yaw.append(np.pi-yaw)
    
    def updatePosition(self, x: float, y: float, yaw: float, p_x: float, p_y: float, p_yaw: float) -> None:
        '''
        hijack the vehicle in order to hit the pedestrian
        '''
        self.record_x.append(-x)
        self.record_y.append(y)
        self.record_yaw.append(np.pi-yaw)
        self.p_x.append(-p_x)
        self.p_y.append(p_y)
        self.p_yaw.append(np.pi-p_yaw)

    def release(self, name):
        print(len(self.record_x))
        for i in range(len(self.record_x)):
            fig = plt.figure()
            # draw vehicle and its history path
            if self.pedestrian == False:
                plt.plot(self.path_x, self.path_y, color='gray')
            plt.plot(self.record_x[0:i], self.record_y[0:i], color='g')
            plt.plot(self.record_x[i], self.record_y[i], marker='.', color='r')
            Arrow(self.record_x[i], self.record_y[i], self.record_yaw[i], 0.2, 'blue')
            # draw pedestrian position and direction
            if self.pedestrian:
                plt.plot(self.p_x[i], self.p_y[i], marker='.', color='yellow')
                Arrow(self.p_x[i], self.p_y[i], self.p_yaw[i], 0.2, 'blue')
            plt.savefig('gif/{}.png'.format(i))
            plt.close()
            self.filenames.append('gif/{}.png'.format(i))
        with imageio.get_writer(f'gif/{name}', mode='I') as writer:
            for filename in self.filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(self.filenames):
            os.remove(filename)

def run():
    global vehicle, world, agent, sensor, walker, dir
    for _ in range(30):
        world.tick()
        time.sleep(0.05)
    time_start=time.time()
    command = 4
    index = 0
    flag = True

    # S curve
    path_x, path_y = getPath()
    path = PATH(path_x, path_y)
    graph = Graph(path_x=path_x, path_y=path_y)
    node = Node(0, 0, 0, 0)
    target_ind = 0
    while(time.time()-time_start <= 3600 and index <= 300):        
        v = vehicle.get_velocity()
        image = image_queue.get()
        location = sensor.get_transform().location
        rotation = sensor.get_transform().rotation
        v_location = vehicle.get_transform().location
        v_rotation = vehicle.get_transform().rotation
        x, y, yaw = v_location.x, v_location.y, v_rotation.yaw/180*np.pi
        graph.add_graph(x, y, yaw)

        node.update(x, y, yaw, math.sqrt(v.x**2 + v.y**2))           
        delta, target_ind = pure_pursuit(node, path, target_ind)

        points = getpoint.getpoints(location.x, location.y, location.z, rotation.pitch, rotation.yaw, rotation.roll, HEIGHT, WIDTH)

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        observations = {}
        observations["bgr"] = image[0:200][75:75+66]
        observations["velocity"] = np.array([v.x,v.y,v.z])
        observations["command"] = command
        observations["points"] = np.array(points)
        observations["direction"] = delta    

        index += 1
        control = agent.run_step(observations)
        print(control.steer)
        vehicle.apply_control(control)       
        keys = getkeys.key_check()
        if "P" in keys:
            break
        time.sleep(0.5)
        world.tick()

    world.tick()
    graph.release('result.gif')

def destroy():
    global actor_list, world, agent, save3rdvout
    for ac in actor_list:
        ac.destroy()
    world.tick()
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    agent.end()
    savebackvout.release()
    saveupvout.release()

def stupid_attack(x, y, pitch, yaw, roll, direction) -> np.ndarray:
    global dir, dir_float
    temp = dir_float - [x, y, pitch, yaw, roll, direction]
    temp = np.array([item[0]**2+item[1]**2+item[2]**2+item[3]**2+item[4]**2+item[5]**2 for item in temp])
    indices = np.argsort(temp)[0:5]
    return indices

if __name__ == "__main__":
    # depict the default image
    cv2.imshow("billboard", np.zeros((480, 640), dtype=np.uint8))
    cv2.waitKey(1)
    init_agent()
    init_utils()
    init_world()
    run()
    destroy()