import math
import numpy as np
import LRHB_spline as LRHBS

def exterior_algebra_max_information_3d(data_point_1,data_point_2,data_point_value,point_delta):

    fit_point = LRHBS.fit_3d(data_point_1,data_point_2,data_point_value,point_delta)

    len_fit_point_1 = len(fit_point[0])
    len_fit_point_2 = len(fit_point[1])

    vector = np.zeros(shape = (3,4))
    max_point = list(np.zeros(2))
    max_weight = 0

    for n1 in range(len_fit_point_1-1):

        p1 = LRHBS.vector_position_find(data_point_1,fit_point[0][n1])

        for n2 in range(len_fit_point_2-1):

            p2 = LRHBS.vector_position_find(data_point_2,fit_point[1][n2])

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

            weight = weight

            if (weight > max_weight):

                max_point[0] = fit_point[0][n1]
                max_point[1] = fit_point[1][n2]

                max_weight = weight

    point_delta_decimal_places_1 = len(str(point_delta[0]).split('.')[1])
    point_delta_decimal_places_2 = len(str(point_delta[1]).split('.')[1])

    max_point[0] = round(max_point[0],point_delta_decimal_places_1)
    max_point[1] = round(max_point[1],point_delta_decimal_places_2)

    return max_point