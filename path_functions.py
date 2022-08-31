import os, math
import numpy as np
import torch
import matplotlib.pyplot as plt


# SMOOTHENING PATH FUNCTIONS:

'''
smoothen_paths()

HOW TO IMPORT:      'from create_paths import smoothen_paths'
WHAT IT DOES:       Takes an array of paths, a start point, and an end point, and returns an array of smoothened path coords.
WEIRD PARAMETERS:   'smooth_val' --> Number of times coordinate values are averaged for "smoother" appearance.
'''

def smoothen_paths(paths, start, goal, smooth_val=5, figsize=(7.5,7.5), save=False, display=False):

    new_paths = []

    # setting up save:
    if save:
        dir_name = "smoothened_paths/"
        if not os.path.exists(dir_name):   # if path folder for this map doesn't exist, create it
            os.mkdir(dir_name)

    for i in range(paths.shape[0]):

        # getting path data from GAN output:
        paths[i] = np.round(paths[i])
        path_coords = np.nonzero(paths[i]==1)

        # sorting and smoothening path_coords array:
        path_coords = __sort_data(path_coords, start, goal)
        new_path_coords = __smoothen(path_coords, smooth_val)

        if save:
            # flatten + write path to file:
            flat_path = new_path_coords.flatten()      # flipped so start point is at front of file, end point is at end of file
            np.savetxt(f"{dir_name}path_{i}.txt", flat_path, fmt='%d') 

        # displaying path data:
        if display:
        
            plt.figure(figsize=figsize)
            plt.imshow(paths[i])
            plt.plot(path_coords[:, 1], path_coords[:, 0], c='r')
            plt.plot(new_path_coords[:, 1], new_path_coords[:, 0], c='g', linewidth=5)
            plt.show()

        # adding sorted and smoothened path coords to new_paths array:
        new_paths.append(new_path_coords)

    return new_paths

'''
__smoothen()

WHAT IT DOES:       Takes an array of path coords and a loop value, then returns an array of smoothened path coords.
WEIRD PARAMETERS:   'loops' --> Number of times coordinate values are averaged for "smoother" appearance.
'''

def __smoothen(path, loops):
    new_path = path.astype(float)

    for _ in range(loops):
        for i in range(1, len(path)-1):

            # get previous, current, and next points:
            prev_x, prev_y = path[i-1][0], path[i-1][1]
            curr_x, curr_y = path[i][0], path[i][1]
            next_x, next_y = path[i+1][0], path[i+1][1]

            # average out values:
            new_path[i] = [(prev_x+curr_x+next_x)/3.0, (prev_y+curr_y+next_y)/3.0]
            
        path = new_path

    return new_path

'''
__sort_data()

WHAT IT DOES:   Takes an image of a path (represented by a 2D array), a start point, and an end point, 
                and attempts to sort the path coordinates found in the image to create an array of path coords in order.
'''

def __sort_data(path, start, goal):

    # ensure start point is proper path start point:
    sorted_path = [[start[0], start[1]]] 

    # create array of points that still need to be sorted:             
    remaining_points = [[x,y] for x,y in path]
    remaining_points.append([goal[0], goal[1]])     # add goal point to remaining points (in case it isn't there already)

    # sorting points:
    sorted_path_len = len(sorted_path)
    while sorted_path[sorted_path_len-1] != [goal[0], goal[1]]:     # break loop once final sorted path point is equal to the goal point

        next_point_idx = None
        smallest_dis = None

        # get last point in array of sorted points:
        prev_x, prev_y = sorted_path[sorted_path_len-1][0], sorted_path[sorted_path_len-1][1]

        for i in range(len(remaining_points)):      # loop through all unsorted points
            # get x and y vals of current unsorted point:
            curr_x, curr_y = remaining_points[i][0], remaining_points[i][1]

            # get distance between last sorted point and current unsorted point:
            dis = math.sqrt((prev_x-curr_x)**2 + (prev_y-curr_y)**2)

            # save index of unsorted point closest to last sorted point:
            if smallest_dis == None or smallest_dis > dis:
                smallest_dis = dis
                next_point_idx = i

        # if not the same point, add unsorted point to sorted array:
        if smallest_dis != 0:           
            sorted_path.append(remaining_points[next_point_idx])

        # remove unsorted point from remaining points array:
        remaining_points.pop(next_point_idx)        

        sorted_path_len = len(sorted_path)      # re-evaluate length of sorted array

    return np.asarray(sorted_path)


# VERIFYING PATH FUNCTIONS:

RADIUS_OF_OBS = 0.5

'''
check_if_avoids_obstacles()

HOW TO IMPORT:      'from create_paths import check_if_avoids_obstacles'
WHAT IT DOES:       Takes a map, an array of x coords, an array of y coords, and a check_dis value, 
                    then returns either 1 (path does not intersect with an obstacle) or 0 (path intersects with an obstacle).
WEIRD PARAMETERS:   'check_dis' --> Step distance that will be traveled along a line connecting two path points when checking for obstacle intersection. 
                    After each step the function re-evaluates if the line passes through an obstacle or not.
'''

def check_if_avoids_obstacles(xs, ys, map, check_dis=0.5):

    # create obstacle_coords array (coords of all points that are non-zero):
    obstacle_coords = torch.fliplr(torch.nonzero(torch.tensor(map))).tolist()

    for idx in range(len(xs)):
        if idx == len(xs)-1:            # checking final point
            if not __check_btw_points([xs[idx],ys[idx]], [xs[idx],ys[idx]], check_dis, obstacle_coords):
                # print("Path intersects obstacles.")
                return 0
        else:                           # checking all other points
            if not __check_btw_points([xs[idx],ys[idx]], [xs[idx+1],ys[idx+1]], check_dis, obstacle_coords):
                # print("Path intersects obstacles.")
                return 0

    return 1 # if none of the prev conditions are met, return 1 (path does not cross an obstacle)

'''
__check_btw_points()

WHAT IT DOES:       Takes two points, a step value, and an array of obstacle coordinates to check if the line between the two points given crosses any obstacles.
WEIRD PARAMETERS:   'step' --> Step distance that will be traveled along the line connecting the two given points when checking for obstacle intersection. 
                    After each step, the function re-evaluates if the line passes through an obstacle or not.
'''

def __check_btw_points(p1, p2, step, obstacle_coords):

    # calc distance between given points:
    dis_btw_points = __calc_distance(p1,p2)
    
    # determining how many steps along line between given points will be taken:
    num_points_to_check = dis_btw_points/step
    if num_points_to_check == 0:
        num_points_to_check = 1     # must check at least the first given coord

    if p2[0]-p1[0] != 0:            # if there is a difference in the x-axis
        
        # calculate step value along x-axis:
        coord_step = (p2[0]-p1[0])/num_points_to_check

        # calcuate equation values:
        m,b = __make_eq(p1, p2)

        for x in np.arange(p1[0], p2[0], coord_step):       # travel along x-axis from start to end by step value
            y = m*x + b             # calculate y

            if not __check_single_point([x,y], obstacle_coords):
                return 0
    
    elif p2[1]-p1[1] != 0:          # if there is a difference in the y-axis (i.e. path with undefined slope)
        
        # calculate step value along y-axis:
        coord_step = (p2[1]-p1[1])/num_points_to_check

        for y in np.arange(p1[1], p2[1], coord_step):       # travel along y-axis from start to end by step value
            x = p1[0]               # determine x

            if not __check_single_point([x,y], obstacle_coords):
                return 0

    else:                           # if there is no difference in either x or y axises (i.e. the same point)
        if not __check_single_point(p1, obstacle_coords):
            return 0

    return 1

'''
__check_single_point()

WHAT IT DOES: Takes a single path point and an array of obstacle coordinates, and uses trigonometry to determine of this specific point is within an obstacle or not.
'''

def __check_single_point(point, obstacle_coords):

    max_obstacle_dis = math.sqrt(RADIUS_OF_OBS**2 + RADIUS_OF_OBS**2)           # this is the furthest away a point could be from an obstacle while still possibly intersecting with it (the centre of the obstacle pixel to a corner)

    for ob_coord in obstacle_coords:            # loop through all obstacles
        dis = __calc_distance(point, ob_coord)

        if (dis <= max_obstacle_dis):           # if distance between current path point and obstacle point is less than the max distance, it could intersect
            # calculate angle btw path point and obstacle point
            theta = math.acos(abs(point[0]-ob_coord[0])/dis)
            # calculate length of obstacle at this angle:
            hypot = RADIUS_OF_OBS/math.cos(theta)
            # if length of obstacle >= distance between current path point and obstacle point --> this point intersects with an obstacle
            if hypot >= dis:
                return 0
    return 1

'''
__calc_distance()

WHAT IT DOES: Takes two points and calculates the distance between them.
'''

def __calc_distance(p1,p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

'''
__make_eq()

WHAT IT DOES: Takes two points and calulates the slope (m) and x-intercept (b) values of the line that connects them.
'''

def __make_eq(p1, p2):
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p1[1]-(m*p1[0])
    return m, b
