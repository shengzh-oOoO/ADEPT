import model
import torch
import torch.nn as nn
import numpy as np
import carla
import cv2
import time
import copy
import pickle
from Regressor import Regressor, Regressor_sig
from torchvision.transforms import GaussianBlur

billboard = np.array([[0, 0], [640, 0], [0, 480], [640, 480]], dtype=np.float32)
Gaussian = GaussianBlur(kernel_size = (3, 3))

class Nvidia_agent():
    def __init__(self,args):
        self.index = 0
        self.perspective_img = None
        self.args = args
        self.steering_model = model.NVIDIA_ORIGIN()
        self.steering_model = torch.load("Nvidia_model/Train_4_NVIDIA_ORIGIN/Epoch8_Val_loss0.00981.pkl")
        self.steering_model.to(self.args.device)
        if self.args.mp4:
            fps = 10
            size = (200,66)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.vout = cv2.VideoWriter()
            self.vout.open(time.strftime('tmp/%Y%m%d-%H-%M-%S.mp4', time.localtime()),fourcc,fps,size,True)

    def run_step(self, observations):
        bgr = observations["bgr"].copy()
        bgr = bgr.astype(np.uint8)
        points = observations["points"].copy()
        for point in points:
            point[1] = point[1] - 75
        
        if self.args.render:
            render_bgr = bgr.copy()
            for point in points:
                cv2.circle(render_bgr, (int(point[0]),int(point[1])), 1, (0,0,255), 4)
            cv2.imshow("render_bgr", render_bgr)
            cv2.waitKey(1)
            cv2.imshow("drive_bgr", bgr)
            cv2.waitKey(1)
        if self.args.mp4:
            self.vout.write(bgr)
        
        
        bgr = np.rollaxis(bgr, 2, 0)
        torch_image = np.expand_dims(bgr, axis = 0)
        torch_image = torch.from_numpy(np.array(torch_image)).float().div_(255)
        torch_image = torch_image.to(self.args.device)
        with torch.no_grad():
            predict_steer = float(self.steering_model(torch_image))
        speed = np.linalg.norm(observations["velocity"])

        if speed < 1:
            throttle = 0.3
            brake = 0
        else:
            throttle = 0.2
            brake = 0
        # throttle = 0.3
        # brake = 0
        control = carla.VehicleControl()
        control.steer = np.clip(predict_steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.manual_gear_shift = False
        return control
    
    def cal_adv(self, observations):
        bgr = observations["bgr"].copy()
        points = observations["points"].copy()
        for point in points:
            point[1] = point[1] - 75

        mask = np.array(np.zeros(bgr.shape), dtype=np.uint8)
        area = np.array([points[0],points[2],points[3],points[1]])
        cv2.fillPoly(mask, [area], (1,1,1))
        mask_ = np.array(np.ones(bgr.shape), dtype=np.uint8)
        cv2.fillPoly(mask_, [area], (0,0,0))
        mask = np.rollaxis(mask, 2, 0)
        mask_ = np.rollaxis(mask_, 2, 0)
        bgr = np.rollaxis(bgr, 2, 0)
        bgr = np.expand_dims(bgr, axis = 0)
        torch_image = np.multiply(bgr, mask_)
        torch_billboard = np.multiply(bgr, mask)
        torch_image = torch.from_numpy(np.array(torch_image)).float().div_(255)
        torch_billboard = torch.from_numpy(np.array(torch_billboard)).float().div_(255)
        torch_image = torch_image.to(self.args.device)
        torch_billboard = torch_billboard.to(self.args.device)
        trans_model = Regressor_sig().to(self.args.device)
        trans_model.load_state_dict(torch.load('fitcolor_model/color_regression_sig.pth'))
        trans_model.eval()
        torch_mask = torch.from_numpy(mask).float().to(self.args.device)           
        direction = observations["direction"]
        target = direction
        target = torch.from_numpy(np.array([[target]])).float()
        target = target.to(self.args.device)
        mseloss = torch.nn.MSELoss()
        learning_rate = 1e-3
        i = -1
        while(1):
            i += 1
            # optimize from the perspective image
            torch_billboard.requires_grad_(True)
            
            trans_torch_billboard = Gaussian.forward(trans_model(torch_billboard.permute(0, 3, 2, 1)).permute(0, 3, 2, 1))         
            predict_steer = self.steering_model(trans_torch_billboard * torch_mask + torch_image)
            loss = mseloss(predict_steer, target)
            loss.backward()
            grad = torch_billboard.grad
            torch_billboard.requires_grad_(False)

            grad = grad * torch_mask
            grad = torch.sign(grad)
            # grad = grad.to(self.args.device)
            torch_billboard = torch_billboard - learning_rate * grad
            torch_billboard[torch_billboard < 0] = 0
            torch_billboard[torch_billboard > 1] = 1

            if(abs(float(predict_steer) - target) < 0.05 or i > 1000):# or abs(torch.sum(grad)) <= 1e-3:
                print(i, "attack:"+str(float(predict_steer)))
                break

        image = (torch_billboard+ torch_image).cpu().numpy()[0]
        image = np.rollaxis(image, 0, 3)
        image = image * 255
        image[image < 0] = 0
        image[image > 255] = 255

        Points = np.asarray(np.array([points[0], points[1], points[2], points[3]]), dtype=np.float32)
        ret = cv2.getPerspectiveTransform(Points, billboard)
        self.perspective_img = cv2.warpPerspective(image, ret, (640, 480), flags=cv2.INTER_AREA).astype(np.uint8)

        cv2.imshow("billboard", self.perspective_img)
        cv2.waitKey(1)
        return observations

    def end(self):
        if (self.args.mp4):
            self.vout.release()