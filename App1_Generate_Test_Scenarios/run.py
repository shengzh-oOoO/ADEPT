import glob
# from mimetypes import init
import os
import sys
# from click import command

# from matplotlib.pyplot import connect
try:
    sys.path.append(glob.glob('../carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# import time
# import math
# import csv

import argparse
import cv2
import numpy as np
import queue

from models import agent_IAs_RL
# import Nvidia_agent

import torch
import random

import getkeys
# import getpoint
import time

client = None
world = None
vehicle = None
vehicle_name = None
agent = None
sensor_id = None
save1thvout = None
save3thvout = None

def connect_carla():
    global client, world, vehicle
    if client == None:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10)
        world = client.get_world()
    return
def init_agent():
    global agent
    parser = argparse.ArgumentParser()
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
    agent = agent_IAs_RL.AgentIAsRL(args)
    return
def find_car():
    global vehicle, world, vehicle_name, sensor_id
    while vehicle == None:
        print("Waiting for the ego vehicle...")
        time.sleep(1)
        possible_vehicles = world.get_actors().filter('vehicle.*')
        for v in possible_vehicles:
            if v.attributes['role_name'] == "hero":
                print("Ego vehicle found")
                vehicle = v
                break
        # for t in world.get_actors().filter('sensor.*'):
        #     if t.id == sensor_id:
        #         t.distory()
        # print(world.get_actors().filter('vehicle.*'))
        # print(world.get_actors().filter('sensor.*'))
    vehicle_name = vehicle.type_id
    return
def show_img(image):
    HEIGHT = 288
    WIDTH = 288
    image = np.array(image.raw_data)
    image = image.reshape((HEIGHT,WIDTH,4))
    image_queue.queue.clear()
    image_queue.put_nowait(image)
    # newimg = cv2.resize(image,dsize=(HEIGHT*2,WIDTH*2))
    # cv2.imshow("network input", newimg)
    # cv2.waitKey(1)
def show_img3(image):
    HEIGHT = 1280
    WIDTH = 1280
    image = np.array(image.raw_data)
    image = image.reshape((HEIGHT,WIDTH,4))
    image_queue3.queue.clear()
    image_queue3.put_nowait(image)
    # newimg = cv2.resize(image,dsize=(HEIGHT*2,WIDTH*2))
    # cv2.imshow("network input", newimg)
    # cv2.waitKey(1)
def set_sensor():
    global image_queue, sensor, vehicle, world, sensor_id, image_queue3, sensor3
    HEIGHT = 288
    WIDTH = 288
    blueprint_library = world.get_blueprint_library()
    image_queue = queue.Queue(1)
    image_queue3 = queue.Queue(1)
    cam_bp = blueprint_library.filter("sensor.camera.rgb")[0]
    cam_bp.set_attribute("image_size_x", "{}".format(HEIGHT))
    cam_bp.set_attribute("image_size_y", "{}".format(WIDTH))
    cam_bp.set_attribute("fov","90")
    spawn_point = carla.Transform(carla.Location(x=1.5,z=2.4))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
    sensor.listen(lambda data: show_img(data))
    sensor_id = sensor.id

    cam_bp = blueprint_library.filter("sensor.camera.rgb")[0]
    cam_bp.set_attribute("image_size_x", "{}".format(1280))
    cam_bp.set_attribute("image_size_y", "{}".format(1280))
    cam_bp.set_attribute("fov","90")
    spawn_point = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-20.0))
    sensor3 = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
    sensor3.listen(lambda data: show_img3(data))
    return
def driving(c):
    global world, vehicle, vehicle_name, image_queue, sensor, save1thvout, save3thvout, image_queue3
    command = c
    while vehicle != None:
        if len(world.get_actors().filter(vehicle_name)) < 1:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
            vehicle = None
            print("END")
            break
        # control = vehicle.get_control()
        # control.throttle = 0.3
        keys = getkeys.key_check()
        # if "A" in keys:
        #     control.steer = -0.5
        # elif "D" in keys:
        #     control.steer = 0.5
        # else:
        #     control.steer = 0
        if "Q" in keys:
            break
        # if("F" in keys):
        #     command = 1
        # elif("H" in keys):
        #     command = 2
        # elif("G" in keys):
        #     command = 3
        # elif("T" in keys):
        #     command = 4
        image = image_queue.get()
        newimg = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        save1thvout.write(newimg)
        cv2.imshow("network input", newimg)
        cv2.waitKey(1)   
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image3 = image_queue3.get()
        newimg3 = cv2.cvtColor(image3, cv2.COLOR_BGRA2BGR)
        save3thvout.write(newimg3)
        newimg3 = cv2.resize(newimg3,(320,320))
        cv2.imshow("3th", newimg3)
        cv2.waitKey(1) 
        v = vehicle.get_velocity()
        observations = {}
        observations["rgb"] = image
        observations["velocity"] = np.array([v.x,v.y,v.z])
        observations["command"] = command
        control = agent.run_step(observations)
        vehicle.apply_control(control)
        print(control)
        time.sleep(0.05)
    return
def flash_agent():
    global agent
    agent.flash()
    return

def init_recoder():
    print("initrecoder")
    global save1thvout, save3thvout
    fps = 30
    size = (288,288)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save1thvout = cv2.VideoWriter()
    save1thvout.open(time.strftime('../../pic/1_%Y%m%d-%H-%M-%S.mp4', time.localtime()),fourcc,fps,size,True)
    size = (1280,1280)
    save3thvout = cv2.VideoWriter()
    save3thvout.open(time.strftime('../../pic/3_%Y%m%d-%H-%M-%S.mp4', time.localtime()),fourcc,fps,size,True)
def destory_recoder():
    print("destory_recoder")
    global save1thvout, save3thvout
    save1thvout.release()
    save3thvout.release()


if __name__ == "__main__":
    # c = input("ego command: LEFT or RIGHT or STRAIGHT or FOLLOW_LANE?")
    c="F"
    if c == "LEFT" or c == "L":
        c = 1
    elif c == "RIGHT" or c == "R":
        c = 2
    elif c == "STRAIGHT" or c == "S":
        c = 3
    else:
        c = 4
    init_agent()
    connect_carla()
    init_recoder()
    while True:
        find_car()
        set_sensor()
        flash_agent()
        driving(c)
        
        keys = getkeys.key_check()
        if "Q" in keys:
            break
    destory_recoder()
    