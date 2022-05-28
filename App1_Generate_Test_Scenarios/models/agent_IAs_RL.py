import numpy as np
import torch
from collections import deque, namedtuple
import cv2
import os

from torch import tensor
import carla
import time

from .model_supervised import Model_Segmentation_Traffic_Light_Supervised
from .model_RL import DQN, Orders

import getkeys

from models import model_supervised

attack_num = 8


class AgentIAsRL:
    def __init__(self, args=None, **kwargs):
        super().__init__(**kwargs)

        self.args = args

        path_to_folder_with_model = args.path_folder_model
        path_to_model_supervised = os.path.join(path_to_folder_with_model, "model_supervised/")
        path_model_supervised = None
        for file in os.listdir(path_to_model_supervised):
            if ".pth" in file:
                if path_model_supervised is not None:
                    raise ValueError(
                        "There is multiple model supervised in folder " +
                        path_to_model_supervised +
                        " you must keep only one!",
                    )
                path_model_supervised = os.path.join(path_to_model_supervised, file)
        if path_model_supervised is None:
            raise ValueError("We didn't find any model supervised in folder " +
                             path_to_model_supervised)

        # All this magic number should match the one used when training supervised...
        self.model_supervised = Model_Segmentation_Traffic_Light_Supervised(
            len(args.steps_image), len(args.steps_image), 1024, 6, 4, args.crop_sky
        )
        self.model_supervised.load_state_dict(
            torch.load(path_model_supervised, map_location=args.device)
        )
        self.model_supervised.to(device=args.device)

        self.encoder = self.model_supervised.encoder
        self.last_conv_downsample = self.model_supervised.last_conv_downsample

        self.action_space = (args.nb_action_throttle + 1) * args.nb_action_steering

        path_to_model_RL = os.path.join(path_to_folder_with_model, "model_RL")
        os.chdir(path_to_model_RL)
        tab_model = []
        for file in os.listdir(path_to_model_RL):
            if ".pth" in file:
                tab_model.append(os.path.join(path_to_model_RL, file))

        if len(tab_model) == 0:
            raise ValueError("We didn't find any RL model in folder "+ path_to_model_RL)

        self.tab_RL_model = []
        for current_model in tab_model:

            current_RL_model = DQN(args, self.action_space).to(device=args.device)
            current_RL_model_dict = current_RL_model.state_dict()

            print("we load RL model ", current_model)
            checkpoint = torch.load(current_model)

            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v
                for k, v in checkpoint["model_state_dict"].items()
                if k in current_RL_model_dict
            }
            # 2. overwrite entries in the existing state dict
            current_RL_model_dict.update(pretrained_dict)
            # 3. load the new state dict
            current_RL_model.load_state_dict(current_RL_model_dict)
            self.tab_RL_model.append(current_RL_model)

        self.window = (
            max([abs(number) for number in args.steps_image]) + 1
        )  # Number of frames to concatenate
        self.RGB_image_buffer = deque([], maxlen=self.window)
        self.device = args.device

        self.state_buffer = deque([], maxlen=self.window)
        self.State = namedtuple("State", ("image", "speed", "order", "steering"))

        if args.crop_sky:
            blank_state = self.State(
                np.zeros(6144, dtype=np.float32), -1, -1, 0
            )  # RGB Image, color channet first for torch
        else:
            blank_state = self.State(np.zeros(8192, dtype=np.float32), -1, -1, 0)
        for _ in range(self.window):
            self.state_buffer.append(blank_state)
            if args.crop_sky:
                self.RGB_image_buffer.append(
                    # np.zeros((3, args.front_camera_height - 120, args.front_camera_width))
                    torch.zeros([3, args.front_camera_height - 120, args.front_camera_width]).unsqueeze(0)
                )
            else:
                self.RGB_image_buffer.append(
                    # np.zeros((3, args.front_camera_height, args.front_camera_width))
                    torch.zeros([3, args.front_camera_height, args.front_camera_width]).unsqueeze(0)
                )

        self.last_steering = 0
        self.last_order = 0

        self.current_timestep = 0

        # if args.mp4:
        #     fps = 30
        #     size = (288,288)
        #     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #     self.vout = cv2.VideoWriter()
        #     self.vout.open(time.strftime('../../tmp/%Y%m%d-%H-%M-%S.mp4', time.localtime()),fourcc,fps,size,True)
        # self.f = open(time.strftime('../../tmp/%Y%m%d-%H-%M-%S.txt', time.localtime()),"w")
        self.hetmap_stat = np.zeros((288,288), dtype=np.float32)
    def flash(self):
        self.window = (
            max([abs(number) for number in self.args.steps_image]) + 1
        )  # Number of frames to concatenate
        self.RGB_image_buffer = deque([], maxlen=self.window)
        self.state_buffer = deque([], maxlen=self.window)
        self.State = namedtuple("State", ("image", "speed", "order", "steering"))

        if self.args.crop_sky:
            blank_state = self.State(
                np.zeros(6144, dtype=np.float32), -1, -1, 0
            )  # RGB Image, color channet first for torch
        else:
            blank_state = self.State(np.zeros(8192, dtype=np.float32), -1, -1, 0)
        for _ in range(self.window):
            self.state_buffer.append(blank_state)
            if self.args.crop_sky:
                self.RGB_image_buffer.append(
                    # np.zeros((3, args.front_camera_height - 120, args.front_camera_width))
                    torch.zeros([3, self.args.front_camera_height - 120, self.args.front_camera_width]).unsqueeze(0)
                )
            else:
                self.RGB_image_buffer.append(
                    # np.zeros((3, args.front_camera_height, args.front_camera_width))
                    torch.zeros([3, self.args.front_camera_height, self.args.front_camera_width]).unsqueeze(0)
                )
        self.last_steering = 0
        self.last_order = 0
        self.current_timestep = 0
        self.hetmap_stat = np.zeros((288,288), dtype=np.float32)
    def act(self, state_buffer, RL_model):
        speeds = []
        order = state_buffer[-1].order
        steerings = []
        for step_image in self.args.steps_image:
            state = state_buffer[step_image + self.window - 1]
            speeds.append(state.speed)
            steerings.append(state.steering)
        images = torch.from_numpy(state_buffer[-1].image).to(self.device, dtype=torch.float32)
        speeds = torch.from_numpy(np.stack(speeds).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )
        steerings = torch.from_numpy(np.stack(steerings).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )
        
        with torch.no_grad():
            quantile_values, _ = RL_model(
                images.unsqueeze(0),
                speeds.unsqueeze(0),
                order,
                steerings.unsqueeze(0),
                self.args.num_quantile_samples,
            )
            return quantile_values.mean(0).argmax(0).item()

    # We had different mapping int/order in our training than in the CARLA benchmark,
    # so we need to remap orders
    def adapt_order(self, incoming_obs_command):
        if incoming_obs_command == 1:  # LEFT
            return Orders.Left.value
        if incoming_obs_command == 2:  # RIGHT
            return Orders.Right.value
        if incoming_obs_command == 3:  # STRAIGHT
            return Orders.Straight.value
        if incoming_obs_command == 4:  # FOLLOW_LANE
            return Orders.Follow_Lane.value

    def cal_control(self, tab_action):
        steer = 0
        throttle = 0
        brake = 0

        for action in tab_action:

            steer += (
                (action % self.args.nb_action_steering) - int(self.args.nb_action_steering / 2)
            ) * (self.args.max_steering / int(self.args.nb_action_steering / 2))
            if action < int(self.args.nb_action_steering * self.args.nb_action_throttle):
                throttle += (int(action / self.args.nb_action_steering)) * (
                    self.args.max_throttle / (self.args.nb_action_throttle - 1)
                )
                brake += 0
            else:
                throttle += 0
                brake += 1.0

        steer = steer / len(tab_action)
        throttle = throttle / len(tab_action)
        if brake < len(tab_action) / 2:
            brake = 0
        else:
            brake = brake / len(tab_action)

        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.manual_gear_shift = False

        return control
    
    def run_step(self, observations):
        self.current_timestep += 1
        rgb = observations["rgb"].copy()
        if self.args.crop_sky:
            rgb = np.array(rgb)[120:, :, :]
        else:
            rgb = np.array(rgb)
        rgb = rgb.astype(np.uint8)
        if self.args.render:
            bgr = rgb[:, :, ::-1]
            bgr = bgr.copy()
            # points = observations["points"]
            points = None
            if(points == None):
                points = [[0,0]]
            for point in points:
                cv2.circle(bgr, (int(point[0]),int(point[1])), 1, (0,0,255), 4)
            
            # cv2.imshow("network input", bgr)
            # cv2.waitKey(1)
        # if self.args.mp4:
        #     bgr = rgb[:, :, ::-1]
        #     self.vout.write(bgr)
        
        # rgb = np.rollaxis(rgb, 2, 0)

        # self.RGB_image_buffer.append(rgb)
        self.RGB_image_buffer.append(torch.from_numpy(np.rollaxis(rgb, 2, 0)).to(dtype=torch.float32).div_(255).unsqueeze(0))

        speed = np.linalg.norm(observations["velocity"])

        order = self.adapt_order(int(observations["command"]))
        if self.last_order != order:
            print("order = ", Orders(order).name)
            self.last_order = order

        # np_array_RGB_input = np.concatenate(
        #     [
        #         self.RGB_image_buffer[indice_image + self.window - 1]
        #         for indice_image in self.args.steps_image
        #     ]
        # )

        # torch_tensor_input = (
        #     torch.from_numpy(np_array_RGB_input)
        #     .to(dtype=torch.float32, device=self.device)
        #     .div_(255)
        #     .unsqueeze(0)
        # )
        
        torch.set_grad_enabled(False)

        torch_array_RGB_input = []
        for indice_image in self.args.steps_image:
            torch_array_RGB_input.append(self.RGB_image_buffer[indice_image + self.window - 1].to(device=self.device))
        torch_tensor_input = torch.cat((torch_array_RGB_input[0],torch_array_RGB_input[1],torch_array_RGB_input[2],torch_array_RGB_input[3]), axis = 1)

        current_encoding_1 = self.encoder(torch_tensor_input)
        # current_encoding_1.requires_grad_(True)
        # current_encoding_1.retain_grad()
        current_encoding_2 = self.last_conv_downsample(current_encoding_1)

        images = current_encoding_2.flatten()
        # with torch.no_grad():
        #     current_encoding = self.encoder(torch_tensor_input)
        #     print(current_encoding.size())
        #     current_encoding = self.last_conv_downsample(current_encoding)
        # print(current_encoding.size())
        current_encoding_np = images.cpu().detach().numpy()

        current_state = self.State(current_encoding_np, speed, order, self.last_steering)
        self.state_buffer.append(current_state)

        speeds = []
        order = self.state_buffer[-1].order
        steerings = []
        for step_image in self.args.steps_image:
            state = self.state_buffer[step_image + self.window - 1]
            speeds.append(state.speed)
            steerings.append(state.steering)
        speeds = torch.from_numpy(np.stack(speeds).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )
        steerings = torch.from_numpy(np.stack(steerings).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )
        tab_action = []
        # min = 0
        for k, RL_model in enumerate(self.tab_RL_model):
            if k >= attack_num:
                break
            # current_action = self.act(self.state_buffer, RL_model)
            quantile_values, _ = RL_model(
                images.unsqueeze(0),
                speeds.unsqueeze(0),
                order,
                steerings.unsqueeze(0),
                self.args.num_quantile_samples,
            )
            current_action = quantile_values.mean(0).argmax(0).item()
            # min += torch.nn.functional.softmax(quantile_values.mean(0), dim = 0)[current_action]
            tab_action.append(current_action)
        # min.backward()
        # grad =  current_encoding_1.grad.cpu().data
        # print(grad.size())
        # np_grad = grad.numpy()[0]
        # np_encoding_1 = current_encoding_1.detach().cpu().numpy()[0]
        # cam = np.zeros((288,288), dtype=np.float32)
        # weights = np.mean(np_grad.reshape([np_grad.shape[0],-1]), axis=1)
        # for i in range(384,512):
        #     w = weights[i]
        #     cam += w * cv2.resize(np_encoding_1[i, :, :],(288,288))
        # cam = np.maximum(cam, 0)
        # cam = cam / cam.max()
        # cam = cv2.resize(cam, (288,288))
        # bgr = rgb[:, :, ::-1]
        # bgr = bgr.copy()
        # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # cam_img = (0.3 * heatmap + 0.7 * bgr).astype(np.uint8)
        # cv2.imshow("cam_img", cam_img)
        # cv2.waitKey(1)
        # print(cam.shape)

        torch.set_grad_enabled(True)

        control = self.cal_control(tab_action)

        self.last_steering = control.steer
        # print("driver:"+str(control))
        # print(tab_action)
        # print("\n")
        # self.f.write("driver:"+str(control)+"\n")
        return control
    
    def cal_adv(self, observations):
        # tmp1 = np.array([i for i in range(0,11)])#13
        # tmp2 = np.array([i for i in range(27,38)])#40
        # tmp3 = np.array([i for i in range(54,65)])#67
        # left_index = np.concatenate([tmp1,tmp2,tmp3])
        # other_index = np.array([i for i in range(0,108)])
        # other_index = np.delete(other_index, left_index)

        rgb = observations["rgb"].copy()
        if self.args.crop_sky:
            rgb = np.array(rgb)[120:, :, :]
        else:
            rgb = np.array(rgb)
        points = observations["points"]
        mask = np.array(np.zeros(rgb.shape), dtype=np.uint8)
        area = np.array([points[0],points[2],points[3],points[1]])
        cv2.fillPoly(mask, [area], (1,1,1))
        mask_ = np.array(np.ones(rgb.shape), dtype=np.uint8)
        cv2.fillPoly(mask_, [area], (0,0,0))

        # cv2.imshow("mask", mask*255)
        # cv2.waitKey(1)
        mask = np.rollaxis(mask, 2, 0)
        mask_ = np.rollaxis(mask_, 2, 0)


        rgb = np.rollaxis(rgb, 2, 0)
        speed = np.linalg.norm(observations["velocity"])
        speeds = [self.state_buffer[1].speed, self.state_buffer[9].speed,self.state_buffer[10].speed, speed]
        order = self.adapt_order(int(observations["command"]))
        steerings = [self.state_buffer[1].steering, self.state_buffer[9].steering,self.state_buffer[10].steering, self.last_steering]
        speeds = torch.from_numpy(np.stack(speeds).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )
        steerings = torch.from_numpy(np.stack(steerings).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )

        np_array_RGB_input = np.concatenate([self.RGB_image_buffer[1],self.RGB_image_buffer[9],self.RGB_image_buffer[10],rgb])
        torch_tensor_input = (
            torch.from_numpy(np_array_RGB_input)
            .to(dtype=torch.float32, device=self.device)
            .div_(255)
            .unsqueeze(0)
        )
        origin_tab_action = []
        with torch.no_grad():
            current_encoding = self.encoder(torch_tensor_input)
            current_encoding = self.last_conv_downsample(current_encoding)
            images = current_encoding.flatten()
            for k, RL_model in enumerate(self.tab_RL_model):
                if k >= attack_num:
                    break
                quantile_values, _ = RL_model(
                    images.unsqueeze(0),
                    speeds.unsqueeze(0),
                    order,
                    steerings.unsqueeze(0),
                    self.args.num_quantile_samples,
                )
                origin_tab_action.append(quantile_values.mean(0).argmax(0).item())
        control = self.cal_control(origin_tab_action)
        print("origin:"+str(control))
        print(origin_tab_action)#如果不做攻击的决策

        target = 93
        # if(speed <= 0.01):
        #     target = 13
        # self.f.write("\n")
        print(self.current_timestep)
        print(speed)
        # self.f.write(str(self.current_timestep)+"\n")
        # self.f.write(str(speed) + "\n")
        # random = np.random.rand(3,288,288)
        # random = np.multiply(random*255, mask)
        # rgb = np.multiply(rgb, mask_)
        # rgb = rgb + random

        # for i in range(10000):
        i = -1
        while(1):
            i+=1
            np_array_RGB_input = np.concatenate([self.RGB_image_buffer[1],self.RGB_image_buffer[9],self.RGB_image_buffer[10],rgb])
            torch_tensor_input = (
                torch.from_numpy(np_array_RGB_input)
                .to(dtype=torch.float32, device=self.device)
                .div_(255)
                .unsqueeze(0)
            )
            torch_tensor_input.requires_grad_(True)
            current_encoding = self.encoder(torch_tensor_input)
            current_encoding = self.last_conv_downsample(current_encoding)
            images = current_encoding.flatten()
            
            # print(images.size())
            # print(speeds.size())
            # print(steerings.size())

            tab_action = []
            
            min = 0
            tmp_steer = 0
            tmp_throttle = 0
            tmp_brake = 0
            
            for k, RL_model in enumerate(self.tab_RL_model):
                if k >= attack_num:
                    break
                quantile_values, _ = RL_model(
                    images.unsqueeze(0),
                    speeds.unsqueeze(0),
                    order,
                    steerings.unsqueeze(0),
                    self.args.num_quantile_samples,
                )
                values = quantile_values.mean(0)
                action = values.argmax(0).item()
                softmax_values = torch.nn.functional.softmax(values,dim = 0)
                # print(softmax_values.size()[0])
                for v in range(0, softmax_values.size()[0]):
                    tmp_steer += ((v % self.args.nb_action_steering) - int(self.args.nb_action_steering / 2)) * (self.args.max_steering / int(self.args.nb_action_steering / 2))*softmax_values[v]
                    if v < int(self.args.nb_action_steering * self.args.nb_action_throttle):
                        tmp_throttle += (int(v / self.args.nb_action_steering)) * (self.args.max_throttle / (self.args.nb_action_throttle - 1))*softmax_values[v]
                    else:
                        tmp_brake += 1.0*softmax_values[v]
                
                # obj1 = torch.max(quantile_values.mean(0)[left_index])
                # obj2 = torch.max(quantile_values.mean(0)[other_index])
                # min -= torch.clamp(obj1 - obj2, max = 0.1)
                
                # _,indices=torch.topk(quantile_values.mean(0),2)
                # min -= torch.clamp(quantile_values.mean(0)[target] - quantile_values.mean(0)[indices[1]], max = 1)
                # if (indices[0]!=target):
                    # min -= torch.nn.functional.softmax(quantile_values.mean(0),dim=0)[target] - torch.nn.functional.softmax(quantile_values.mean(0),dim=0)[indices[0]]
                    # min -= quantile_values.mean(0)[target] - quantile_values.mean(0)[indices[0]]
                # else:
                    # min -= torch.nn.functional.softmax(quantile_values.mean(0),dim=0)[target] - torch.nn.functional.softmax(quantile_values.mean(0),dim=0)[indices[1]]
                # min -= torch.nn.functional.softmax(quantile_values.mean(0),dim=0)[target]
                # min -= quantile_values.mean(0)[target]

                tab_action.append(action)

            
            

            tmp_steer = tmp_steer / attack_num
            # min = torch.pow((tmp_steer - (-0.2)),2)
            min = tmp_steer
            control = self.cal_control(tab_action)

            if(control.steer < -0.11):
                print("epoch\t"+str(i)+"\t"+str(min.data.cpu().numpy())+"\t"+str(tab_action)+"\t"+ str(tmp_steer.data.cpu().numpy()))
                print(control)
                # self.f.write("epoch\t"+str(i)+"\t"+str(min.data.cpu().numpy())+"\t"+str(tab_action)+"\n")
                advimg = rgb.copy()
                advimg = np.rollaxis(advimg,0,3)
                advimg = advimg.astype(np.uint8)
                bgr = advimg[:, :, ::-1]
                cv2.imshow("advimg", bgr)
                cv2.waitKey(1)
                break

            min.backward()
            grad =  torch_tensor_input.grad.cpu().data

            g = np.multiply(grad[0][9:12].numpy(), mask)

            eps = 1e-1

            rgb = rgb - eps * g * 255
            rgb[rgb < 0] = 0
            rgb[rgb > 255] = 255

            if(i%100 == 0):
                print("epoch\t"+str(i)+"\t"+str(min.data.cpu().numpy())+"\t"+str(tab_action)+"\t"+ str(tmp_steer.data.cpu().numpy()))
                print(self.cal_control(tab_action))
                self.f.write("epoch\t"+str(i)+"\t"+str(min.data.cpu().numpy())+"\t"+str(tab_action)+"\n")
                advimg = rgb.copy()
                advimg = np.rollaxis(advimg,0,3)
                advimg = advimg.astype(np.uint8)
                bgr = advimg[:, :, ::-1]
                cv2.imshow("advimg", bgr)
                cv2.waitKey(1)
            keys = getkeys.key_check()
            if "P" in keys:
                break
        
        np_array_RGB_input = np.concatenate([self.RGB_image_buffer[1],self.RGB_image_buffer[9],self.RGB_image_buffer[10],rgb])
        torch_tensor_input = (
            torch.from_numpy(np_array_RGB_input)
            .to(dtype=torch.float32, device=self.device)
            .div_(255)
            .unsqueeze(0)
        )
        tab_action = []
        with torch.no_grad():
            current_encoding = self.encoder(torch_tensor_input)
            current_encoding = self.last_conv_downsample(current_encoding)
            images = current_encoding.flatten()
            for k, RL_model in enumerate(self.tab_RL_model):
                if k >= attack_num:
                    break
                quantile_values, _ = RL_model(
                    images.unsqueeze(0),
                    speeds.unsqueeze(0),
                    order,
                    steerings.unsqueeze(0),
                    self.args.num_quantile_samples,
                )
                tab_action.append(quantile_values.mean(0).argmax(0).item())
        
        control = self.cal_control(tab_action)
        print("attack:"+str(control))
        # self.f.write("attack:"+str(control)+"\n")
        print(tab_action)#攻击完成后，预计的决策

        rgb = np.rollaxis(rgb,0,3)
        observations["rgb"] = rgb
        return observations
    def end(self):
        # if (self.args.mp4):
        #     self.vout.release()
        # self.f.close()
        return