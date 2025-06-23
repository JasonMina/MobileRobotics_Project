import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math


#Black_Upper = np.array([0, 0, 0])
#Black_Lower = np.array([180, 255, 58]]) use yellow
Lower_Blue = np.array([90, 50, 70])
Upper_Blue = np.array([128, 255, 255])
Yellow_Lower = np.array([15, 50, 20])
Yellow_Upper = np.array([35, 255, 255]) 
Lower_Green = np.array([36, 50, 70])
Upper_Green = np.array([89, 255, 255])
Red_Lower = np.array([159, 50, 70])
Red_Upper = np.array([180, 255, 255])
Corner_Areas = 350
Map_Width = 1400
Map_Height = 1000
Predefined_Points= [[0,0],[0,0],[0,0],[0,0]] #for now, test it on setup one time then change
obstacle_width = 100
obstacle_height= 100 #check and need of change of unit
Thymio_Radius = 140

id_thymio = 1
rtag_l = 115


def find_center(Shapes, Compare_Area):
  Points = []
  for C in Shapes:
    if cv2.contourArea(C) > Compare_Area:
        M = cv2.moments(C)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        Points += [[cX,cY]]
  return Points


def Find_Contours(mask, lower, upper, Compare_Area):
    mask=cv2.inRange(mask,lower,upper,Compare_Area)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8),iterations=2)
    shapes, u = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    List=[]
    
    for c in shapes:
        if (cv2.contourArea(c) > Compare_Area):
                List+=[c]
    
    return List


def Find_direction(pt1, pt2, pt3):
    d12 = [math.dist(pt1, pt2), 1, 2]
    d13 = [math.dist(pt1, pt3) , 1, 3]
    d23 = [math.dist(pt2, pt3) , 2, 3]
    dmin= min([d12[0], d13[0], d23[0]])
    if dmin in d12:
        vertex=pt3
        mid= [(pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2]
    elif dmin in d13:
        vertex=pt2
        mid= [(pt1[0]+pt3[0])/2,(pt1[1]+pt3[1])/2]
    else:
        vertex=pt1
        mid= [(pt3[0]+pt2[0])/2,(pt3[1]+pt2[1])/2]
    direction = [vertex[0]-mid[0],vertex[1]-mid[1]]
    return direction, mid, vertex


def Order_Corner_pts(Corner_Points):
    Corner_Points=sorted(Corner_Points, key=lambda x: (int(x[1]), int(x[0]))) 
    if Corner_Points[1][0] < Corner_Points[0][0] :
        Corner_Points[1], Corner_Points[0] = Corner_Points[0], Corner_Points[1]
    if Corner_Points[3][0] < Corner_Points[2][0] :
        Corner_Points[3], Corner_Points[2] = Corner_Points[2], Corner_Points[3]
    return Corner_Points


def Warpit(image,Corner_Points):
    Origin_Points = np.float32(Corner_Points)
    Destination_Points= np.float32([[0,0],[Map_Width, 0], [0,Map_Height], [Map_Width, Map_Height]])
    Transform = cv2.getPerspectiveTransform(Origin_Points, Destination_Points)
    return cv2.warpPerspective(image, Transform, (Map_Width, Map_Height))


def Birds_Eye_init(vid):
    global Predefined_Points

    while(1):
        ret, image = vid.read()

        Image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        Image_filtered = cv2.bilateralFilter(Image_HSV, 9, 80, 80)
        Detect_Black = cv2.inRange(Image_filtered, Yellow_Lower, Yellow_Upper)
        Corners, u = cv2.findContours(Detect_Black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Corner_Points = find_center(Corners, Corner_Areas)
        if len(Corner_Points) == 4 :
            break
    
    Predefined_Points = Order_Corner_pts(Corner_Points)
    Birds_Eye_Image = Warpit(image, Predefined_Points)
    return Birds_Eye_Image

def Birds_Eye(image):
    global Predefined_Points
    Birds_Eye_Image = Warpit(image,Predefined_Points)
    return Birds_Eye_Image


def Find_Goals(Birds_eye):
    HSV_Birds_eye = cv2.cvtColor(Birds_eye, cv2.COLOR_BGR2HSV)
    Goal_candidates =  Find_Contours(HSV_Birds_eye, Lower_Green, Upper_Green, Corner_Areas)
    [Goal_center1, Goal_center2] = find_center(Goal_candidates, Corner_Areas)
    if cv2.contourArea(Goal_candidates[1])>cv2.contourArea(Goal_candidates[0]):
        First_Goal = Goal_center1
        Second_Goal = Goal_center2
    else:
        First_Goal = Goal_center2
        Second_Goal = Goal_center1
    return First_Goal, Second_Goal


def Find_Obstacles(Birds_eye):
    HSV_Birds_eye = cv2.cvtColor(Birds_eye, cv2.COLOR_BGR2HSV)
    Obstacle_candidates =  Find_Contours(HSV_Birds_eye, Lower_Blue, Upper_Blue, Corner_Areas)
    Obstacle_Centers = find_center(Obstacle_candidates, Corner_Areas)
    return Obstacle_Centers


def Detect_Obstacles(Obstacle_Centers,obstacle_width,obstacle_height, Thymio_Radius):
    Obstacle_List=[]
    for i in range(len(Obstacle_Centers)):
        x1=Obstacle_Centers[i][0] - (obstacle_width/2)#-Thymio_Radius/2
        y1=Obstacle_Centers[i][1] + (obstacle_height/2)#+Thymio_Radius/2
        x2=Obstacle_Centers[i][0] + (obstacle_width/2)#+Thymio_Radius/2
        y2=Obstacle_Centers[i][1] + (obstacle_height/2)#+Thymio_Radius/2
        x3=Obstacle_Centers[i][0] - (obstacle_width/2)#-Thymio_Radius/2
        y3=Obstacle_Centers[i][1] - (obstacle_height/2)#-Thymio_Radius/2
        x4=Obstacle_Centers[i][0] + (obstacle_width/2)#+Thymio_Radius/2
        y4=Obstacle_Centers[i][1] - (obstacle_height/2)#-Thymio_Radius/2
        Obstacle_List += [[[int(x1),int(y1)],[int(x2),int(y2)],[int(x4),int(y4)],[int(x3),int(y3)]]]
    return Obstacle_List


def Enlarge_Obstacles(Obstacle_Centers,obstacle_width,obstacle_height, Thymio_Radius):
    Obstacle_List=[]
    for i in range(len(Obstacle_Centers)):
        x1=Obstacle_Centers[i][0] - (obstacle_width/2)-Thymio_Radius/2
        y1=Obstacle_Centers[i][1] + (obstacle_height/2)+Thymio_Radius/2
        x2=Obstacle_Centers[i][0] + (obstacle_width/2)+Thymio_Radius/2
        y2=Obstacle_Centers[i][1] + (obstacle_height/2)+Thymio_Radius/2
        x3=Obstacle_Centers[i][0] - (obstacle_width/2)-Thymio_Radius/2
        y3=Obstacle_Centers[i][1] - (obstacle_height/2)-Thymio_Radius/2
        x4=Obstacle_Centers[i][0] + (obstacle_width/2)+Thymio_Radius/2
        y4=Obstacle_Centers[i][1] - (obstacle_height/2)-Thymio_Radius/2
        Obstacle_List += [[[int(x1),int(y1)],[int(x2),int(y2)],[int(x4),int(y4)],[int(x3),int(y3)]]]
    return Obstacle_List


def find_thymio(Birds_eye):
    gray = cv2.cvtColor(Birds_eye, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    corners2 = np.array([c[0] for c in corners])

    thymio_center = [0,0]
    thymio_angle = 0
    is_vision = 0

    for idx, i in enumerate(ids):
        if i == id_thymio:
            thymio_center, thymio_angle = find_thymio_position(corners2[idx])
            is_vision = True
            break

    return thymio_center, thymio_angle, is_vision

def find_thymio_position(corners):
    bot_mid = (corners[0] + corners[1])/2
    mid = [corners[:, 0].mean(), corners[:, 1].mean()]
    thymio_center = (bot_mid + mid)/2
    thymio_angle = np.arctan2(mid[1]-bot_mid[1],  mid[0]-bot_mid[0])

    return thymio_center, thymio_angle