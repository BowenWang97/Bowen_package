import math
import numpy as np
import LRHB_spline as LRHBS

def LRHBS_sample_point_add_3d(data_point_1,data_point_2,new_point,level):

    if (level == 0):

        if (new_point[0] not in data_point_1[0][0]):

            i1 = LRHBS.position_find(data_point_1[0][0],new_point[0])

            data_point_1[0][0].insert(i1+1,new_point[0])

        if (new_point[1] not in data_point_2[0][0]):

            i2 = LRHBS.position_find(data_point_2[0][0],new_point[1])
        
            data_point_2[0][0].insert(i2+1,new_point[1])

    elif (level+1 > len(data_point_1)):

        data_point_1.append([new_point[0]])
        data_point_2.append([new_point[1]])

    else:

        position_1 = LRHBS.vector_position_find(data_point_1,new_point[0])

        if (position_1[2] == level):

            i1 = position_1[0]
            b1 = position_1[1]

        else:

            i1 = position_1[3]
            b1 = position_1[4]

        if (new_point[0] not in data_point_1[level][b1]):

            data_point_1[level][b1].insert(i1,new_point[0])

        position_2 = LRHBS.vector_position_find(data_point_2,new_point[1])

        if (position_2[2] == level):

            i2 = position_2[0]
            b2 = position_2[1]

        else:

            i2 = position_2[3]
            b2 = position_2[4]

        if (new_point[1] not in data_point_2[level][b2]):

            data_point_2[level][b2].insert(i2,new_point[1])