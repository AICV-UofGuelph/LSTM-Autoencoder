import os, math
import numpy as np
import torch
import matplotlib.pyplot as plt


# SMOOTHENING PATH FUNCTIONS:

'''
HOW TO IMPORT: 'from create_paths import smoothen paths'
WHAT IT DOES: Takes an array of paths, a start point, and an end point, and returns an array of smoothened path coords.
PARAMETERS: smooth_val --> 
'''

def smoothen_paths(paths, start, goal, smooth_val=5, figsize=(7.5,7.5), save=False, display=False):

    new_paths = []

    if save:
        dir_name = "smoothened_paths/"
        if not os.path.exists(dir_name):                                    # if path folder for this map doesn't exist, create it
            os.mkdir(dir_name)

    for i in range(paths.shape[0]):

        # getting path data from GAN output:
        paths[i] = np.round(paths[i])
        path_coords = np.nonzero(paths[i]==1)
        path_coords = sort_data(path_coords, start, goal)
        new_path_coords = smoothen(path_coords, smooth_val)

        if save:
            # flatten + write path to file:
            flat_path = new_path_coords.flatten()      # flipped so start point is at front of file, end point is at end of file
            np.savetxt(f"{dir_name}path_{i}.txt", flat_path, fmt='%d') 

        if display:
            # displaying path data:
            plt.figure(figsize=figsize)
            plt.imshow(paths[i])
            plt.plot(path_coords[:, 1], path_coords[:, 0], c='r')
            plt.plot(new_path_coords[:, 1], new_path_coords[:, 0], c='g', linewidth=5)
            plt.show()

        new_paths.append(new_path_coords)

    return new_paths

def smoothen(path, loops):
    new_path = path.astype(float)

    for _ in range(loops):
        for i in range(1, len(path)-1):
            prev_x, prev_y = path[i-1][0], path[i-1][1]
            curr_x, curr_y = path[i][0], path[i][1]
            next_x, next_y = path[i+1][0], path[i+1][1]

            new_path[i] = [(prev_x+curr_x+next_x)/3.0, (prev_y+curr_y+next_y)/3.0]
            
        path = new_path

    return new_path

def sort_data(path, start, goal):

    sorted_path = [[start[0], start[1]]]              # ensure start point is proper path start point
    remaining_points = [[x,y] for x,y in path]
    remaining_points.append([goal[0], goal[1]])

    sorted_path_len = len(sorted_path)
    while sorted_path[sorted_path_len-1] != [goal[0], goal[1]]:
        prev_x, prev_y = sorted_path[sorted_path_len-1][0], sorted_path[sorted_path_len-1][1]
        next_point_idx = None
        smallest_dis = None
        
        for i in range(len(remaining_points)):
            curr_x, curr_y = remaining_points[i][0], remaining_points[i][1]
            dis = math.sqrt((prev_x-curr_x)**2 + (prev_y-curr_y)**2)

            if smallest_dis == None or smallest_dis > dis:
                smallest_dis = dis
                next_point_idx = i

        if smallest_dis != 0:
            sorted_path.append(remaining_points[next_point_idx])
        remaining_points.pop(next_point_idx)

        sorted_path_len = len(sorted_path)

    return np.asarray(sorted_path)


# VERIFYING PATH FUNCTIONS:

def check_if_avoids_obstacles(xs, ys, map, check_dis=0.5):

    # create obstacle_coords array:
    obstacle_coords = torch.fliplr(torch.nonzero(torch.tensor(map))).tolist()

    # check if any points are on an obstacle or if any paths connecting points crosses an obstacle:
    radius_of_obs = 0.5

    for idx in range(len(xs)):
        if idx == len(xs)-1:            # checking final point
            if not check_btw_points([xs[idx],ys[idx]], [xs[idx],ys[idx]], check_dis, obstacle_coords, radius_of_obs):
                # print("Path intersects obstacles.")
                return 0
        else:                           # checking all other points
            if not check_btw_points([xs[idx],ys[idx]], [xs[idx+1],ys[idx+1]], check_dis, obstacle_coords, radius_of_obs):
                # print("Path intersects obstacles.")
                return 0

    return 1 # if none of the prev conditions are met, return 1 (path does not cross an obstacle)


def check_btw_points(p1, p2, step, obstacle_coords, radius_of_obs):
    dis_btw_points = calc_distance(p1,p2)
    
    num_points_to_check = dis_btw_points/step
    if num_points_to_check == 0:
        num_points_to_check = 1

    if p2[0]-p1[0] != 0:            # if there is a difference in the x-axis
        
        coord_step = (p2[0]-p1[0])/num_points_to_check
        m,b = make_eq(p1, p2)

        for x in np.arange(p1[0], p2[0], coord_step):
            y = m*x + b

            if not detailed_check([x,y], obstacle_coords, radius_of_obs):
                return 0
    
    elif p2[1]-p1[1] != 0:          # if there is a difference in the y-axis (i.e. path with undefined slope)
        
        coord_step = (p2[1]-p1[1])/num_points_to_check

        for y in np.arange(p1[1], p2[1], coord_step):
            x = p1[0]

            if not detailed_check([x,y], obstacle_coords, radius_of_obs):
                return 0

    else:                           # if there is no difference in either x or y axises (i.e. the same point)
        if not detailed_check(p1, obstacle_coords, radius_of_obs):
            return 0

    return 1


def detailed_check(point, obstacle_coords, radius_of_obs):
    max_obstacle_dis = math.sqrt(radius_of_obs**2 + radius_of_obs**2)

    for ob_coord in obstacle_coords:
        dis = calc_distance(point, ob_coord)

        if (dis <= max_obstacle_dis):
            # calculate angle btw this point and obstacle point
            theta = math.acos(abs(point[0]-ob_coord[0])/dis)
            # calculate length of hypotenuse
            hypot = radius_of_obs/math.cos(theta)
            # if hypotenuse >= dis --> this point intersects with an obstacle
            if hypot >= dis:
                return 0
    return 1


def calc_distance(p1,p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def make_eq(p1, p2):
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p1[1]-(m*p1[0])
    return m, b
