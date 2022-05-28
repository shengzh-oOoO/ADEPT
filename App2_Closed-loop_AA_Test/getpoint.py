from numpy.core.fromnumeric import ptp
import pandas as pd
import math
import numpy as np

# H = 288
# W = 288
# Points = [[-4.1,162.5,7.55], [-10.1,162.5,7.55], [-4.1,162.5,3.05], [-10.1,162.5,3.05]]
Points = [[-6.9,166.1,4.9], [-10.1,166.1,4.9], [-6.9,166.1,2.5], [-10.1,166.1,2.5]]
F_vector = [1,0,0]
L_vector = [0,1,0]
U_vector = [0,0,1]
def trans_pitch(x,y,z,pitch):
    newx = x*math.cos(pitch) - z*math.sin(pitch)
    newy = y
    newz = x*math.sin(pitch) + z*math.cos(pitch)
    return newx, newy, newz
def trans_yaw(x,y,z,yaw):
    newx = x*math.cos(yaw) - y*math.sin(yaw)
    newy = x*math.sin(yaw) + y*math.cos(yaw)
    newz = z
    return newx,newy,newz
def trans_roll(x,y,z,roll):
    newx = x
    newy = z*math.sin(roll) + y*math.cos(roll)
    newz = z*math.cos(roll) - y*math.sin(roll)
    return newx,newy,newz

def trans_vector(vector, pitch, yaw, roll):
    x,y,z = vector[0],vector[1],vector[2]
    x,y,z = trans_pitch(x,y,z,pitch)
    x,y,z = trans_yaw(x,y,z,yaw)
    x,y,z = trans_roll(x,y,z,roll)
    return [x,y,z]

def getpoints(x,y,z,pitch, yaw, roll, H, W):
    pitch, yaw, roll = pitch/180*math.pi, yaw/180*math.pi, roll/180*math.pi
    Fvec = trans_vector(F_vector, pitch, yaw, roll)
    Lvec = trans_vector(L_vector, pitch, yaw, roll)
    Uvec = trans_vector(U_vector, pitch, yaw, roll)
    result = []
    for i in range(4):
        Point = Points[i]
        Point_vector = [Point[0]-x, Point[1]-y, Point[2]-z]
        F = np.dot(Point_vector,Fvec)
        X = np.dot(Point_vector,Lvec)
        Y = np.dot(Point_vector,Uvec)
        tan_X = X / F
        tan_Y = Y / F
        if(F < 0):
            return None
        px = (tan_X * 0.5 * W) + W/2
        py = -(tan_Y * 0.5 * H) + H/2
        result.append([round(px),round(py)])
    return result