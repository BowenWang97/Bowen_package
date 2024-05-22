import math
import numpy as np
import LRHB_spline as LRHBS

def exterior_algebra_max_information_3d(data_point_1,data_point_2,data_point_value,point_delta,level):

    fit_point = LRHBS.fit_3d(data_point_1,data_point_2,data_point_value,point_delta)

    point_delta_decimal_places_1 = len(str(point_delta[0]).split('.')[1])
    point_delta_decimal_places_2 = len(str(point_delta[1]).split('.')[1])

    len_fit_point_1 = len(fit_point[0])
    len_fit_point_2 = len(fit_point[1])

    vector = np.zeros(shape = (3,4))
    max_point = list(np.zeros(2))
    max_weight = 0

    if (level == 0):

        for n1 in range(len_fit_point_1-1):

            p1 = LRHBS.vector_position_find_2d(data_point_1,fit_point[0][n1])

            for n2 in range(len_fit_point_2-1):

                p2 = LRHBS.vector_position_find_2d(data_point_2,fit_point[1][n2])

                if (p1[2] >= p1[5]):

                    level = p1[2]
                    block = p1[1]
                    r1 = p1[0]
                    l1 = p1[0]+1

                    if (p1[2] >= p1[5]):

                        r2 = p2[0]
                        l2 = p2[0]+1

                    else:

                        r2 = p2[3]-1
                        l2 = p2[3]

                else:

                    level = p1[5]
                    block = p1[4]
                    r1 = p1[3]-1
                    l1 = p1[3]

                    if (p1[2] >= p1[5]):

                        r2 = p2[0]
                        l2 = p2[0]+1

                    else:

                        r2 = p2[3]-1
                        l2 = p2[3]

                weight = 0

                vector[0][0] = fit_point[0][n1]-data_point_1[p1[2]][p1[1]][p1[0]]
                vector[0][1] = vector[0][0]
                vector[0][2] = fit_point[0][n1]-data_point_1[p1[5]][p1[4]][p1[3]]
                vector[0][3] = vector[0][2]
                vector[1][0] = fit_point[1][n2]-data_point_2[p2[2]][p2[1]][p2[0]]
                vector[1][1] = fit_point[1][n2]-data_point_2[p2[5]][p2[4]][p2[3]]
                vector[1][2] = vector[1][0]
                vector[1][3] = vector[1][1]
                vector[2][0] = fit_point[2][n1][n2]-data_point_value[level][block][r1][r2]
                vector[2][1] = fit_point[2][n1][n2]-data_point_value[level][block][r1][l2]
                vector[2][2] = fit_point[2][n1][n2]-data_point_value[level][block][l1][r2]
                vector[2][3] = fit_point[2][n1][n2]-data_point_value[level][block][l1][l2]

                matrix = np.zeros(shape = (3,3))

                for vector_1 in range(2):

                    for d in range(3):

                        matrix[d][0] = vector[d][vector_1]

                    for vector_2 in range(vector_1+1,3):

                        for d in range(3):

                            matrix[d][1] = vector[d][vector_2]

                        for vector_3 in range(vector_2+1,4):

                            for d in range(3):

                                matrix[d][2] = vector[d][vector_3]

                            weight = weight+abs(np.linalg.det(matrix))

                weight = weight*(fit_point[0][n1]-data_point_1[p1[2]][p1[1]][p1[0]])*(fit_point[0][n1]-data_point_1[p1[5]][p1[4]][p1[3]])*(fit_point[1][n2]-data_point_2[p1[2]][p1[1]][p1[0]])*(fit_point[1][n2]-data_point_2[p1[5]][p1[4]][p1[3]])

                if (weight > max_weight):

                    max_point[0] = fit_point[0][n1]
                    max_point[1] = fit_point[1][n2]

                    max_weight = weight

    else:
        
        for n1 in range(len_fit_point_1-1):

            p1 = LRHBS.vector_position_find_2d(data_point_1,fit_point[0][n1])

            for l in range(len(data_point_1)):

                for b in range(len(data_point_1[l])):

                    if (round(fit_point[0][n1],point_delta_decimal_places_1) in data_point_1[l][b]):

                        condition_1 = True

                        break

                    else:

                        condition_1 = False

                if (condition_1):

                    break

            if (condition_1):

                loop_1 = False

            else:

                loop_1 = True

            for n2 in range(len_fit_point_2-1):

                p2 = LRHBS.vector_position_find_2d(data_point_2,fit_point[1][n2])

                for l in range(len(data_point_2)):

                    for b in range(len(data_point_2[l])):

                        if (round(fit_point[1][n2],point_delta_decimal_places_2) in data_point_2[l][b]):

                            condition_2 = True

                            break

                        else:

                            condition_2 = False

                    if (condition_2):

                        break

                if (condition_2):

                    loop_2 = False

                else:

                    loop_2 = True

                if (loop_1 and loop_2):

                    data_point_value_position = LRHBS.vector_position_find_3d(data_point_1,data_point_2,[fit_point[0][n1],fit_point[1][n2]])

                    level = data_point_value_position[0]
                    block = data_point_value_position[1]
                    r1 = data_point_value_position[2]
                    l1 = data_point_value_position[3]
                    r2 = data_point_value_position[4]
                    l2 = data_point_value_position[5]

                    weight = 0

                    vector[0][0] = fit_point[0][n1]-data_point_1[p1[2]][p1[1]][p1[0]]
                    vector[0][1] = vector[0][0]
                    vector[0][2] = fit_point[0][n1]-data_point_1[p1[5]][p1[4]][p1[3]]
                    vector[0][3] = vector[0][2]
                    vector[1][0] = fit_point[1][n2]-data_point_2[p2[2]][p2[1]][p2[0]]
                    vector[1][1] = fit_point[1][n2]-data_point_2[p2[5]][p2[4]][p2[3]]
                    vector[1][2] = vector[1][0]
                    vector[1][3] = vector[1][1]
                    vector[2][0] = fit_point[2][n1][n2]-data_point_value[level][block][r1][r2]
                    vector[2][1] = fit_point[2][n1][n2]-data_point_value[level][block][r1][l2]
                    vector[2][2] = fit_point[2][n1][n2]-data_point_value[level][block][l1][r2]
                    vector[2][3] = fit_point[2][n1][n2]-data_point_value[level][block][l1][l2]

                    matrix = np.zeros(shape = (3,3))

                    for vector_1 in range(2):

                        for d in range(3):

                            matrix[d][0] = vector[d][vector_1]

                        for vector_2 in range(vector_1+1,3):

                            for d in range(3):

                                matrix[d][1] = vector[d][vector_2]

                            for vector_3 in range(vector_2+1,4):

                                for d in range(3):

                                    matrix[d][2] = vector[d][vector_3]

                                weight = weight+abs(np.linalg.det(matrix))

                    weight = weight*(fit_point[0][n1]-data_point_1[p1[2]][p1[1]][p1[0]])*(fit_point[0][n1]-data_point_1[p1[5]][p1[4]][p1[3]])*(fit_point[1][n2]-data_point_2[p1[2]][p1[1]][p1[0]])*(fit_point[1][n2]-data_point_2[p1[5]][p1[4]][p1[3]])

                    if (weight > max_weight):

                        max_point[0] = fit_point[0][n1]
                        max_point[1] = fit_point[1][n2]

                        max_weight = weight

    max_point[0] = round(max_point[0],point_delta_decimal_places_1)
    max_point[1] = round(max_point[1],point_delta_decimal_places_2)

    return max_point

def LRHBS_sample_point_add_3d(data_point_1,data_point_2,new_point,point_delta,level):

    data_point_value_position = LRHBS.vector_position_find_3d(data_point_1,data_point_2,new_point)

    l = data_point_value_position[0]
    b = data_point_value_position[1]
    l1 = data_point_value_position[2]
    l2 = data_point_value_position[4]

    # loop_status = True

    # while (loop_status):

    #     if (level > data_point_value_position[0]):

    #         if (data_point_value_position[2] < 2 or data_point_value_position[3] > len(data_point_1[l][b])-3 or (data_point_value_position[4] < 2 or data_point_value_position[5] > len(data_point_2[l][b])-3)):
                
    #             point_delta_decimal_places_1 = len(str(point_delta[0]*10).split('.')[1])
    #             point_delta_decimal_places_2 = len(str(point_delta[1]*10).split('.')[1])

    #             new_point[0] = round(new_point[0],point_delta_decimal_places_1)
    #             new_point[1] = round(new_point[1],point_delta_decimal_places_2)

    #             data_point_value_position = LRHBS.vector_position_find_3d(data_point_1,data_point_2,new_point)

    #             l = data_point_value_position[0]
    #             b = data_point_value_position[1]
    #             l1 = data_point_value_position[2]
    #             l2 = data_point_value_position[4]

    #             level = level-1

    #         else:

    #             loop_status = False

    #     else:

    #         loop_status = False

    if (level > data_point_value_position[0]):

        if (level == len(data_point_1)-1):

            for b in range(len(data_point_1[level])):

                if (new_point[0] < data_point_1[level][b][0]):

                    b0 = b
                    break

                else:

                    b0 = len(data_point_1[level])

            data_point_1[level].insert(b0,[new_point[0]])
            data_point_2[level].insert(b0,[new_point[1]])

        else:

            data_point_1.append([[new_point[0]]])
            data_point_2.append([[new_point[1]]])

    else:

        if (new_point[0] not in data_point_1[l][b]):

            if (l == 0):

                data_point_1[l][b].insert(l1+1,new_point[0])

            else:

                data_point_1[l][b].insert(l1,new_point[0])

        if (new_point[1] not in data_point_2[l][b]):

            if (l == 0):

                data_point_2[l][b].insert(l2+1,new_point[1])

            else:

                data_point_2[l][b].insert(l2,new_point[1])