from pyvisgraph import *
from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
import math



def list_to_point(coord_obstacles):
    point_all_obstacles = []
    for i in range(len(coord_obstacles)):
        point_obstacle = []
        for j in range(len(coord_obstacles[i])):
            point_obstacle.append(Point(coord_obstacles[i][j][0], coord_obstacles[i][j][1]))
        point_all_obstacles.append(point_obstacle)
    return point_all_obstacles

def point_to_list(point_obstacles):
    list_obstacles = []
    for i in range(len(point_obstacles)):
        list_obstacles.append([point_obstacles[i].x, point_obstacles[i].y])
    return list_obstacles


def err_angle(x, y, theta, x1, y1):
    return - theta + np.arctan2((y1 - y), (x1 - x))


def distance_to_goal(x, y, x1, y1):
    d_x = abs(x1 - x)
    d_y = abs(y1 - y)
    return (math.sqrt(d_x ** 2 + d_y ** 2))

def compute_path(start, goal, coord_obstacles):
    point_obstacles = list_to_point(coord_obstacles)
    path = VisGraph()
    path.build(point_obstacles)
    point_goals = path.shortest_path(Point(start[0], start[1]), Point(goal[0], goal[1]))
    list_goals = point_to_list(point_goals)
    return list_goals

def next_goal(thymio_center, goals, current_goal):
    if distance_to_goal(thymio_center[0], thymio_center[1], goals[current_goal][0], goals[current_goal][1]) < 40:
        if goals[current_goal] == goals[-1]:
            return -1
        else:
            current_goal = current_goal + 1
            return current_goal

    else :
        return current_goal
