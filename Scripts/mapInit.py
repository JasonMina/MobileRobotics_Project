import vision
import gnav
import cv2
import matplotlib.pyplot as plt
import numpy as np


def map_init(vid):
    birds_eye_image = vision.Birds_Eye_init(vid)

    Birds_Eye_image_HSV = cv2.cvtColor(birds_eye_image, cv2.COLOR_BGR2HSV)
    Detect_Blue = cv2.inRange(Birds_Eye_image_HSV, vision.Lower_Blue, vision.Upper_Blue)
    Detect_Green = cv2.inRange(Birds_Eye_image_HSV, vision.Lower_Green, vision.Upper_Green)
 
    thymio_center, thymio_angle, is_vision = vision.find_thymio(birds_eye_image)

    cv2.arrowedLine(birds_eye_image, (int(thymio_center[0]), int(thymio_center[1])),
                    ( int(thymio_center[0] + 100*np.cos(thymio_angle)), int(thymio_center[1] + 100*np.sin(thymio_angle)) ),
                    (255, 255, 255), 10)
    cv2.circle(birds_eye_image, (int(thymio_center[0]), int(thymio_center[1])), 5, (255, 255, 255), 5)

    First_Goal, Second_Goal = vision.Find_Goals(birds_eye_image)
    cv2.circle(birds_eye_image, (First_Goal[0], First_Goal[1]), 5, (0, 255, 0), 5)
    cv2.circle(birds_eye_image, (Second_Goal[0], Second_Goal[1]), 5, (0, 255, 0), 5)

    Obstacle_Centers = vision.Find_Obstacles(birds_eye_image)
    Obstacle_List = vision.Detect_Obstacles(Obstacle_Centers, vision.obstacle_width, vision.obstacle_height, vision.Thymio_Radius)
    for i in range(4):
        cv2.circle(birds_eye_image, (Obstacle_List[i][0]), 5, (80, 0, 0), 5)
        cv2.circle(birds_eye_image, (Obstacle_List[i][1]), 5, (80, 0, 0), 5)
        cv2.circle(birds_eye_image, (Obstacle_List[i][2]), 5, (80, 0, 0), 5)
        cv2.circle(birds_eye_image, (Obstacle_List[i][3]), 5, (80, 0, 0), 5)
    
    Dilated_Obstacles = vision.Enlarge_Obstacles(Obstacle_Centers, vision.obstacle_width, 
                                                 vision.obstacle_height, vision.Thymio_Radius)
    for i in range(4):
        cv2.rectangle(birds_eye_image, (Dilated_Obstacles[i][3]), (Dilated_Obstacles[i][1]), (255, 0, 0), -1)
    Final_MAP_HSV = cv2.cvtColor(birds_eye_image, cv2.COLOR_BGR2HSV)
    cv2.inRange(Final_MAP_HSV, vision.Lower_Blue, vision.Upper_Blue)

    cv2.circle(birds_eye_image, (First_Goal[0], First_Goal[1]), 5, (255, 255, 255), 5)
    cv2.circle(birds_eye_image, (Second_Goal[0], Second_Goal[1]), 10, (255, 255, 255), 10)

    


    goals = [First_Goal, Second_Goal]
    list_goals1 = gnav.compute_path(thymio_center, goals[0], Dilated_Obstacles)
    list_goals2 = gnav.compute_path(goals[0], goals[1], Dilated_Obstacles)

    list_goals1.pop()

    list_goals = []
    for element in list_goals1:
        list_goals.append(element)
    for element in list_goals2:
        list_goals.append(element)

    for i in range(1,len(list_goals)):
        cv2.line(birds_eye_image, [int(list_goals[i-1][0]), int(list_goals[i-1][1])],
             [int(list_goals[i][0]), int(list_goals[i][1])], (255, 0, 255), 5)
    cv2.imwrite("pathimage.png",birds_eye_image)
    plt.imshow(birds_eye_image)
    plt.title("Final_Map")

    return thymio_center, thymio_angle, list_goals