import math
import numpy as np
import B_spline as BS

def exterior_algebra_max_to_zero_2d(data_point,point_delta,order=3):

    fit_point = BS.fit_2d(data_point,point_delta,order)

    size_fit_point = fit_point.shape
    len_fit_point = size_fit_point[1]

    vector = np.zeros(shape = (2,2))
    weight_0 = 0

    for n in range(len_fit_point-1):

        i = BS.position_find(data_point[0],fit_point[0][n])

        vector[0][0] = fit_point[0][n]-data_point[0][i]
        vector[0][1]= fit_point[0][n]-data_point[0][i+1]
        vector[1][0] = fit_point[1][n]-data_point[1][i]
        vector[1][1]= fit_point[1][n]-data_point[1][i+1]

        weight = -abs(vector[0][0]*vector[1][1]-vector[0][1]*vector[1][0])/fit_point[1][n]

        if (weight > weight_0):

            max_point = fit_point[0][n]

            weight_0 = weight

    return max_point

def exterior_algebra_max_to_zero_3d(data_point,data_point_value,point_delta,order=[3,3]):

    fit_point = BS.fit_3d(data_point,data_point_value,point_delta,order)

    len_fit_point_1 = len(fit_point[0])
    len_fit_point_2 = len(fit_point[1])

    vector = np.zeros(shape = (3,4))
    vector_0 = np.zeros(shape = (3,4))
    max_point = np.zeros(2)
    max_weight = 0

    for n1 in range(len_fit_point_1-1):

        i1 = BS.position_find(data_point[0],fit_point[0][n1])

        for n2 in range(len_fit_point_2-1):

            i2 = BS.position_find(data_point[1],fit_point[1][n2])

            weight = 0

            vector[0][0] = fit_point[0][n1]-data_point[0][i1]
            vector[0][1] = vector[0][0]
            vector[0][2] = fit_point[0][n1]-data_point[0][i1+1]
            vector[0][3] = vector[0][2]
            vector[1][0] = fit_point[1][n2]-data_point[1][i2]
            vector[1][1] = fit_point[1][n2]-data_point[1][i2+1]
            vector[1][2] = vector[1][0]
            vector[1][3] = vector[1][1]
            vector[2][0] = fit_point[2][n1][n2]-data_point_value[i1][i2]
            vector[2][1] = fit_point[2][n1][n2]-data_point_value[i1][i2+1]
            vector[2][2] = fit_point[2][n1][n2]-data_point_value[i1+1][i2]
            vector[2][3] = fit_point[2][n1][n2]-data_point_value[i1+1][i2+1]

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

            weight = -weight/fit_point[2][n1][n2]

            if (weight > max_weight):

                max_point[0] = fit_point[0][n1]
                max_point[1] = fit_point[1][n2]

                max_weight = weight

    return max_point

def exterior_algebra_max_to_zero_4d(data_point,data_point_value,point_delta,order=[3,3,3]):

    fit_point = BS.fit_4d(data_point,data_point_value,point_delta,order)

    len_fit_point_1 = len(fit_point[0])
    len_fit_point_2 = len(fit_point[1])
    len_fit_point_3 = len(fit_point[2])

    vector = np.zeros(shape = (4,8))
    max_point = np.zeros(3)
    weight_0 = 0

    for n1 in range(len_fit_point_1-1):

        i1 = BS.position_find(data_point[0],fit_point[0][n1])

        for n2 in range(len_fit_point_2-1):

            i2 = BS.position_find(data_point[1],fit_point[1][n2])

            for n3 in range(len_fit_point_3-1):

                i3 = BS.position_find(data_point[2],fit_point[2][n3])

                weight = 0

                vector[0][0] = fit_point[0][n1]-data_point[0][i1]
                vector[0][1] = vector[0][0]
                vector[0][2] = vector[0][0]
                vector[0][3] = vector[0][0]
                vector[0][4] = fit_point[0][n1]-data_point[0][i1+1]
                vector[0][5] = vector[0][4]
                vector[0][6] = vector[0][4]
                vector[0][7] = vector[0][4]
                vector[1][0] = fit_point[1][n2]-data_point[1][i2]
                vector[1][1] = vector[1][0]
                vector[1][2] = fit_point[1][n2]-data_point[1][i2+1]
                vector[1][3] = vector[1][2]
                vector[1][4] = vector[1][0]
                vector[1][5] = vector[1][0]
                vector[1][6] = vector[1][2]
                vector[1][7] = vector[1][2]
                vector[2][0] = fit_point[2][n3]-data_point[2][i3]
                vector[2][1] = fit_point[2][n3]-data_point[2][i3+1]
                vector[2][2] = vector[2][0]
                vector[2][3] = vector[2][1]
                vector[2][4] = vector[2][0]
                vector[2][5] = vector[2][1]
                vector[2][6] = vector[2][0]
                vector[2][7] = vector[2][1]
                vector[3][0] = fit_point[3][n1][n2][n3]-data_point_value[i1][i2][i3]
                vector[3][1] = fit_point[3][n1][n2][n3]-data_point_value[i1][i2][i3+1]
                vector[3][2] = fit_point[3][n1][n2][n3]-data_point_value[i1][i2+1][i3]
                vector[3][3] = fit_point[3][n1][n2][n3]-data_point_value[i1][i2+1][i3+1]
                vector[3][4] = fit_point[3][n1][n2][n3]-data_point_value[i1+1][i2][i3]
                vector[3][5] = fit_point[3][n1][n2][n3]-data_point_value[i1+1][i2][i3+1]
                vector[3][6] = fit_point[3][n1][n2][n3]-data_point_value[i1+1][i2+1][i3]
                vector[3][7] = fit_point[3][n1][n2][n3]-data_point_value[i1+1][i2+1][i3+1]

                matrix = np.zeros(shape = (4,4))

                for vector_1 in range(5):

                    for d in range(4):

                        matrix[d][0] = vector[d][vector_1]

                    for vector_2 in range(vector_1+1,6):

                        for d in range(4):

                            matrix[d][1] = vector[d][vector_2]

                        for vector_3 in range(vector_2+1,7):

                            for d in range(4):

                                matrix[d][2] = vector[d][vector_3]

                            for vector_4 in range(vector_3+1,8):

                                for d in range(4):

                                    matrix[d][3] = vector[d][vector_4]

                                weight = weight+abs(np.linalg.det(matrix))

                weight = -weight/fit_point[3][n1][n2][n3]

                if (weight > weight_0):

                    max_point[0] = fit_point[0][n1]
                    max_point[1] = fit_point[1][n2]
                    max_point[2] = fit_point[2][n3]

                    weight_0 = weight

    return max_point

def exterior_algebra_max_to_zero_and_max_to_inf_4d(data_point,data_point_value_1,data_point_value_2,point_delta,order=[3,3,3]):

    fit_point_1 = BS.fit_4d(data_point,data_point_value_1,point_delta,order)
    fit_point_2 = BS.fit_4d(data_point,data_point_value_2,point_delta,order)

    len_fit_point_1 = len(fit_point_1[0])
    len_fit_point_2 = len(fit_point_1[1])
    len_fit_point_3 = len(fit_point_1[2])

    max_point = np.zeros(3)
    weight_0 = 0

    for n1 in range(len_fit_point_1-1):

        i1 = BS.position_find(data_point[0],fit_point_1[0][n1])

        for n2 in range(len_fit_point_2-1):

            i2 = BS.position_find(data_point[1],fit_point_1[1][n2])

            for n3 in range(len_fit_point_3-1):

                i3 = BS.position_find(data_point[2],fit_point_1[2][n3])

                weight = 0

                vector = np.zeros(shape = (4,8))
                weight_1 = 0

                vector[0][0] = fit_point_1[0][n1]-data_point[0][i1]
                vector[0][1] = vector[0][0]
                vector[0][2] = vector[0][0]
                vector[0][3] = vector[0][0]
                vector[0][4] = fit_point_1[0][n1]-data_point[0][i1+1]
                vector[0][5] = vector[0][4]
                vector[0][6] = vector[0][4]
                vector[0][7] = vector[0][4]
                vector[1][0] = fit_point_1[1][n2]-data_point[1][i2]
                vector[1][1] = vector[1][0]
                vector[1][2] = fit_point_1[1][n2]-data_point[1][i2+1]
                vector[1][3] = vector[1][2]
                vector[1][4] = vector[1][0]
                vector[1][5] = vector[1][0]
                vector[1][6] = vector[1][2]
                vector[1][7] = vector[1][2]
                vector[2][0] = fit_point_1[2][n3]-data_point[2][i3]
                vector[2][1] = fit_point_1[2][n3]-data_point[2][i3+1]
                vector[2][2] = vector[2][0]
                vector[2][3] = vector[2][1]
                vector[2][4] = vector[2][0]
                vector[2][5] = vector[2][1]
                vector[2][6] = vector[2][0]
                vector[2][7] = vector[2][1]
                vector[3][0] = fit_point_1[3][n1][n2][n3]-data_point_value_1[i1][i2][i3]
                vector[3][1] = fit_point_1[3][n1][n2][n3]-data_point_value_1[i1][i2][i3+1]
                vector[3][2] = fit_point_1[3][n1][n2][n3]-data_point_value_1[i1][i2+1][i3]
                vector[3][3] = fit_point_1[3][n1][n2][n3]-data_point_value_1[i1][i2+1][i3+1]
                vector[3][4] = fit_point_1[3][n1][n2][n3]-data_point_value_1[i1+1][i2][i3]
                vector[3][5] = fit_point_1[3][n1][n2][n3]-data_point_value_1[i1+1][i2][i3+1]
                vector[3][6] = fit_point_1[3][n1][n2][n3]-data_point_value_1[i1+1][i2+1][i3]
                vector[3][7] = fit_point_1[3][n1][n2][n3]-data_point_value_1[i1+1][i2+1][i3+1]

                matrix = np.zeros(shape = (4,4))

                for vector_1 in range(5):

                    for d in range(4):

                        matrix[d][0] = vector[d][vector_1]

                    for vector_2 in range(vector_1+1,6):

                        for d in range(4):

                            matrix[d][1] = vector[d][vector_2]

                        for vector_3 in range(vector_2+1,7):

                            for d in range(4):

                                matrix[d][2] = vector[d][vector_3]

                            for vector_4 in range(vector_3+1,8):

                                for d in range(4):

                                    matrix[d][3] = vector[d][vector_4]

                                weight_1 = weight_1+abs(np.linalg.det(matrix))

                weight_1 = -weight_1/fit_point_1[3][n1][n2][n3]\
                
                vector = np.zeros(shape = (4,8))
                weight_2 = 0

                vector[0][0] = fit_point_2[0][n1]-data_point[0][i1]
                vector[0][1] = vector[0][0]
                vector[0][2] = vector[0][0]
                vector[0][3] = vector[0][0]
                vector[0][4] = fit_point_2[0][n1]-data_point[0][i1+1]
                vector[0][5] = vector[0][4]
                vector[0][6] = vector[0][4]
                vector[0][7] = vector[0][4]
                vector[1][0] = fit_point_2[1][n2]-data_point[1][i2]
                vector[1][1] = vector[1][0]
                vector[1][2] = fit_point_2[1][n2]-data_point[1][i2+1]
                vector[1][3] = vector[1][2]
                vector[1][4] = vector[1][0]
                vector[1][5] = vector[1][0]
                vector[1][6] = vector[1][2]
                vector[1][7] = vector[1][2]
                vector[2][0] = fit_point_2[2][n3]-data_point[2][i3]
                vector[2][1] = fit_point_2[2][n3]-data_point[2][i3+1]
                vector[2][2] = vector[2][0]
                vector[2][3] = vector[2][1]
                vector[2][4] = vector[2][0]
                vector[2][5] = vector[2][1]
                vector[2][6] = vector[2][0]
                vector[2][7] = vector[2][1]
                vector[3][0] = fit_point_2[3][n1][n2][n3]-data_point_value_2[i1][i2][i3]
                vector[3][1] = fit_point_2[3][n1][n2][n3]-data_point_value_2[i1][i2][i3+1]
                vector[3][2] = fit_point_2[3][n1][n2][n3]-data_point_value_2[i1][i2+1][i3]
                vector[3][3] = fit_point_2[3][n1][n2][n3]-data_point_value_2[i1][i2+1][i3+1]
                vector[3][4] = fit_point_2[3][n1][n2][n3]-data_point_value_2[i1+1][i2][i3]
                vector[3][5] = fit_point_2[3][n1][n2][n3]-data_point_value_2[i1+1][i2][i3+1]
                vector[3][6] = fit_point_2[3][n1][n2][n3]-data_point_value_2[i1+1][i2+1][i3]
                vector[3][7] = fit_point_2[3][n1][n2][n3]-data_point_value_2[i1+1][i2+1][i3+1]

                matrix = np.zeros(shape = (4,4))

                for vector_1 in range(5):

                    for d in range(4):

                        matrix[d][0] = vector[d][vector_1]

                    for vector_2 in range(vector_1+1,6):

                        for d in range(4):

                            matrix[d][1] = vector[d][vector_2]

                        for vector_3 in range(vector_2+1,7):

                            for d in range(4):

                                matrix[d][2] = vector[d][vector_3]

                            for vector_4 in range(vector_3+1,8):

                                for d in range(4):

                                    matrix[d][3] = vector[d][vector_4]

                                weight_2 = weight_2+abs(np.linalg.det(matrix))

                weight_2 = weight_2*np.exp(fit_point_2[3][n1][n2][n3]/100)

                weight = weight_1+weight_2

                if (weight > weight_0):

                    max_point[0] = fit_point_1[0][n1]
                    max_point[1] = fit_point_1[1][n2]
                    max_point[2] = fit_point_1[2][n3]

                    weight_0 = weight

    return max_point

def exterior_algebra_max_to_zero_5d(data_point,data_point_value,point_delta,order=3):

    fit_point = BS.fit_5d(data_point,data_point_value,point_delta,order)

    len_fit_point_1 = len(fit_point[0])
    len_fit_point_2 = len(fit_point[1])
    len_fit_point_3 = len(fit_point[2])
    len_fit_point_4 = len(fit_point[3])

    vector = np.zeros(shape = (5,16))
    max_point = np.zeros(4)
    weight_0 = 0

    for n1 in range(len_fit_point_1-1):

        i1 = BS.position_find(data_point[0],fit_point[0][n1])

        for n2 in range(len_fit_point_2-1):

            i2 = BS.position_find(data_point[1],fit_point[1][n2])

            for n3 in range(len_fit_point_3-1):

                i3 = BS.position_find(data_point[2],fit_point[2][n3])

                for n4 in range(len_fit_point_4-1):

                    i4 = BS.position_find(data_point[3],fit_point[3][n4])

                    weight = 0

                    vector[0][0] = fit_point[0][n1]-data_point[0][i1]
                    vector[0][1] = vector[0][0]
                    vector[0][2] = vector[0][0]
                    vector[0][3] = vector[0][0]
                    vector[0][4] = vector[0][0]
                    vector[0][5] = vector[0][0]
                    vector[0][6] = vector[0][0]
                    vector[0][7] = vector[0][0]
                    vector[0][8] = fit_point[0][n1]-data_point[0][i1+1]
                    vector[0][9] = vector[0][8]
                    vector[0][10] = vector[0][8]
                    vector[0][11] = vector[0][8]
                    vector[0][12] = vector[0][8]
                    vector[0][13] = vector[0][8]
                    vector[0][14] = vector[0][8]
                    vector[0][15] = vector[0][8]
                    vector[1][0] = fit_point[1][n2]-data_point[1][i2]
                    vector[1][1] = vector[1][0]
                    vector[1][2] = vector[1][0]
                    vector[1][3] = vector[1][0]
                    vector[1][4] = fit_point[1][n2]-data_point[1][i2+1]
                    vector[1][5] = vector[1][4]
                    vector[1][6] = vector[1][4]
                    vector[1][7] = vector[1][4]
                    vector[1][8] = vector[1][0]
                    vector[1][9] = vector[1][0]
                    vector[1][10] = vector[1][0]
                    vector[1][11] = vector[1][0]
                    vector[1][12] = vector[1][4]
                    vector[1][13] = vector[1][4]
                    vector[1][14] = vector[1][4]
                    vector[1][15] = vector[1][4]
                    vector[2][0] = fit_point[2][n3]-data_point[2][i3]
                    vector[2][1] = vector[2][0]
                    vector[2][2] = fit_point[2][n3]-data_point[2][i3+1]
                    vector[2][3] = vector[2][1]
                    vector[2][4] = vector[2][0]
                    vector[2][5] = vector[2][0]
                    vector[2][6] = vector[2][1]
                    vector[2][7] = vector[2][1]
                    vector[2][8] = vector[2][0]
                    vector[2][9] = vector[2][0]
                    vector[2][10] = vector[2][1]
                    vector[2][11] = vector[2][1]
                    vector[2][12] = vector[2][0]
                    vector[2][13] = vector[2][0]
                    vector[2][14] = vector[2][1]
                    vector[2][15] = vector[2][1]
                    vector[3][0] = fit_point[3][n4]-data_point[3][i4]
                    vector[3][1] = fit_point[3][n4]-data_point[3][i4+1]
                    vector[3][2] = vector[3][0]
                    vector[3][3] = vector[3][1]
                    vector[3][4] = vector[3][0]
                    vector[3][5] = vector[3][1]
                    vector[3][6] = vector[3][0]
                    vector[3][7] = vector[3][1]
                    vector[3][8] = vector[3][0]
                    vector[3][9] = vector[3][1]
                    vector[3][10] = vector[3][0]
                    vector[3][11] = vector[3][1]
                    vector[3][12] = vector[3][0]
                    vector[3][13] = vector[3][1]
                    vector[3][14] = vector[3][0]
                    vector[3][15] = vector[3][1]
                    vector[4][0] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2][i3][i4]
                    vector[4][1] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2][i3][i4+1]
                    vector[4][2] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2][i3+1][i4]
                    vector[4][3] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2][i3+1][i4+1]
                    vector[4][4] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2+1][i3][i4]
                    vector[4][5] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2+1][i3][i4+1]
                    vector[4][6] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2+1][i3+1][i4]
                    vector[4][7] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2+1][i3+1][i4+1]
                    vector[4][8] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2][i3][i4]
                    vector[4][9] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2][i3][i4+1]
                    vector[4][10] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2][i3+1][i4]
                    vector[4][11] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2][i3+1][i4+1]
                    vector[4][12] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2+1][i3][i4]
                    vector[4][13] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2+1][i3][i4+1]
                    vector[4][14] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2+1][i3+1][i4]
                    vector[4][15] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2+1][i3+1][i4+1]

                    matrix = np.zeros(shape = (5,5))

                    for vector_1 in range(12):

                        for d in range(5):

                            matrix[d][0] = vector[d][vector_1]

                        for vector_2 in range(vector_1+1,13):

                            for d in range(5):

                                matrix[d][1] = vector[d][vector_2]

                            for vector_3 in range(vector_2+1,14):

                                for d in range(5):

                                    matrix[d][2] = vector[d][vector_3]

                                for vector_4 in range(vector_3+1,15):

                                    for d in range(5):

                                        matrix[d][3] = vector[d][vector_4]

                                    for vector_5 in range(vector_4+1,16):

                                        for d in range(5):

                                            matrix[d][4] = vector[d][vector_5]

                                        weight = weight+abs(np.linalg.det(matrix))

                    weight = -weight/fit_point[4][n1][n2][n3][n4]

                    if (weight > weight_0):

                        max_point[0] = fit_point[0][n1]
                        max_point[1] = fit_point[1][n2]
                        max_point[2] = fit_point[2][n3]
                        max_point[3] = fit_point[3][n4]

                        weight_0 = weight

    return max_point

def exterior_algebra_max_information_4d(data_point,data_point_value,point_delta,order=[3,3,3]):

    fit_point = BS.fit_4d(data_point,data_point_value,point_delta,order)

    len_fit_point_1 = len(fit_point[0])
    len_fit_point_2 = len(fit_point[1])
    len_fit_point_3 = len(fit_point[2])

    vector = np.zeros(shape = (4,8))
    max_point = np.zeros(3)
    weight_0 = 0

    for n1 in range(len_fit_point_1-1):

        i1 = BS.position_find(data_point[0],fit_point[0][n1])

        for n2 in range(len_fit_point_2-1):

            i2 = BS.position_find(data_point[1],fit_point[1][n2])

            for n3 in range(len_fit_point_3-1):

                i3 = BS.position_find(data_point[2],fit_point[2][n3])

                weight = 0

                vector[0][0] = fit_point[0][n1]-data_point[0][i1]
                vector[0][1] = vector[0][0]
                vector[0][2] = vector[0][0]
                vector[0][3] = vector[0][0]
                vector[0][4] = fit_point[0][n1]-data_point[0][i1+1]
                vector[0][5] = vector[0][4]
                vector[0][6] = vector[0][4]
                vector[0][7] = vector[0][4]
                vector[1][0] = fit_point[1][n2]-data_point[1][i2]
                vector[1][1] = vector[1][0]
                vector[1][2] = fit_point[1][n2]-data_point[1][i2+1]
                vector[1][3] = vector[1][2]
                vector[1][4] = vector[1][0]
                vector[1][5] = vector[1][0]
                vector[1][6] = vector[1][2]
                vector[1][7] = vector[1][2]
                vector[2][0] = fit_point[2][n3]-data_point[2][i3]
                vector[2][1] = fit_point[2][n3]-data_point[2][i3+1]
                vector[2][2] = vector[2][0]
                vector[2][3] = vector[2][1]
                vector[2][4] = vector[2][0]
                vector[2][5] = vector[2][1]
                vector[2][6] = vector[2][0]
                vector[2][7] = vector[2][1]
                vector[3][0] = fit_point[3][n1][n2][n3]-data_point_value[i1][i2][i3]
                vector[3][1] = fit_point[3][n1][n2][n3]-data_point_value[i1][i2][i3+1]
                vector[3][2] = fit_point[3][n1][n2][n3]-data_point_value[i1][i2+1][i3]
                vector[3][3] = fit_point[3][n1][n2][n3]-data_point_value[i1][i2+1][i3+1]
                vector[3][4] = fit_point[3][n1][n2][n3]-data_point_value[i1+1][i2][i3]
                vector[3][5] = fit_point[3][n1][n2][n3]-data_point_value[i1+1][i2][i3+1]
                vector[3][6] = fit_point[3][n1][n2][n3]-data_point_value[i1+1][i2+1][i3]
                vector[3][7] = fit_point[3][n1][n2][n3]-data_point_value[i1+1][i2+1][i3+1]

                matrix = np.zeros(shape = (4,4))

                for vector_1 in range(5):

                    for d in range(4):

                        matrix[d][0] = vector[d][vector_1]

                    for vector_2 in range(vector_1+1,6):

                        for d in range(4):

                            matrix[d][1] = vector[d][vector_2]

                        for vector_3 in range(vector_2+1,7):

                            for d in range(4):

                                matrix[d][2] = vector[d][vector_3]

                            for vector_4 in range(vector_3+1,8):

                                for d in range(4):

                                    matrix[d][3] = vector[d][vector_4]

                                weight = weight+abs(np.linalg.det(matrix))

                weight = weight

                if (weight > weight_0):

                    max_point[0] = fit_point[0][n1]
                    max_point[1] = fit_point[1][n2]
                    max_point[2] = fit_point[2][n3]

                    weight_0 = weight

    return max_point

def exterior_algebra_max_information_5d(data_point,data_point_value,point_delta,order=[3,3,3]):

    fit_point = BS.fit_5d(data_point,data_point_value,point_delta,order)

    len_fit_point_1 = len(fit_point[0])
    len_fit_point_2 = len(fit_point[1])
    len_fit_point_3 = len(fit_point[2])
    len_fit_point_4 = len(fit_point[3])

    vector = np.zeros(shape = (5,16))
    max_point = np.zeros(4)
    weight_0 = 0

    for n1 in range(len_fit_point_1-1):

        i1 = BS.position_find(data_point[0],fit_point[0][n1])

        for n2 in range(len_fit_point_2-1):

            i2 = BS.position_find(data_point[1],fit_point[1][n2])

            for n3 in range(len_fit_point_3-1):

                i3 = BS.position_find(data_point[2],fit_point[2][n3])

                for n4 in range(len_fit_point_4-1):

                    i4 = BS.position_find(data_point[3],fit_point[3][n4])

                    weight = 0

                    vector[0][0] = fit_point[0][n1]-data_point[0][i1]
                    vector[0][1] = vector[0][0]
                    vector[0][2] = vector[0][0]
                    vector[0][3] = vector[0][0]
                    vector[0][4] = vector[0][0]
                    vector[0][5] = vector[0][0]
                    vector[0][6] = vector[0][0]
                    vector[0][7] = vector[0][0]
                    vector[0][8] = fit_point[0][n1]-data_point[0][i1+1]
                    vector[0][9] = vector[0][8]
                    vector[0][10] = vector[0][8]
                    vector[0][11] = vector[0][8]
                    vector[0][12] = vector[0][8]
                    vector[0][13] = vector[0][8]
                    vector[0][14] = vector[0][8]
                    vector[0][15] = vector[0][8]
                    vector[1][0] = fit_point[1][n2]-data_point[1][i2]
                    vector[1][1] = vector[1][0]
                    vector[1][2] = vector[1][0]
                    vector[1][3] = vector[1][0]
                    vector[1][4] = fit_point[1][n2]-data_point[1][i2+1]
                    vector[1][5] = vector[1][4]
                    vector[1][6] = vector[1][4]
                    vector[1][7] = vector[1][4]
                    vector[1][8] = vector[1][0]
                    vector[1][9] = vector[1][0]
                    vector[1][10] = vector[1][0]
                    vector[1][11] = vector[1][0]
                    vector[1][12] = vector[1][4]
                    vector[1][13] = vector[1][4]
                    vector[1][14] = vector[1][4]
                    vector[1][15] = vector[1][4]
                    vector[2][0] = fit_point[2][n3]-data_point[2][i3]
                    vector[2][1] = vector[2][0]
                    vector[2][2] = fit_point[2][n3]-data_point[2][i3+1]
                    vector[2][3] = vector[2][1]
                    vector[2][4] = vector[2][0]
                    vector[2][5] = vector[2][0]
                    vector[2][6] = vector[2][1]
                    vector[2][7] = vector[2][1]
                    vector[2][8] = vector[2][0]
                    vector[2][9] = vector[2][0]
                    vector[2][10] = vector[2][1]
                    vector[2][11] = vector[2][1]
                    vector[2][12] = vector[2][0]
                    vector[2][13] = vector[2][0]
                    vector[2][14] = vector[2][1]
                    vector[2][15] = vector[2][1]
                    vector[3][0] = fit_point[3][n4]-data_point[3][i4]
                    vector[3][1] = fit_point[3][n4]-data_point[3][i4+1]
                    vector[3][2] = vector[3][0]
                    vector[3][3] = vector[3][1]
                    vector[3][4] = vector[3][0]
                    vector[3][5] = vector[3][1]
                    vector[3][6] = vector[3][0]
                    vector[3][7] = vector[3][1]
                    vector[3][8] = vector[3][0]
                    vector[3][9] = vector[3][1]
                    vector[3][10] = vector[3][0]
                    vector[3][11] = vector[3][1]
                    vector[3][12] = vector[3][0]
                    vector[3][13] = vector[3][1]
                    vector[3][14] = vector[3][0]
                    vector[3][15] = vector[3][1]
                    vector[4][0] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2][i3][i4]
                    vector[4][1] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2][i3][i4+1]
                    vector[4][2] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2][i3+1][i4]
                    vector[4][3] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2][i3+1][i4+1]
                    vector[4][4] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2+1][i3][i4]
                    vector[4][5] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2+1][i3][i4+1]
                    vector[4][6] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2+1][i3+1][i4]
                    vector[4][7] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1][i2+1][i3+1][i4+1]
                    vector[4][8] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2][i3][i4]
                    vector[4][9] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2][i3][i4+1]
                    vector[4][10] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2][i3+1][i4]
                    vector[4][11] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2][i3+1][i4+1]
                    vector[4][12] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2+1][i3][i4]
                    vector[4][13] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2+1][i3][i4+1]
                    vector[4][14] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2+1][i3+1][i4]
                    vector[4][15] = fit_point[4][n1][n2][n3][n4]-data_point_value[i1+1][i2+1][i3+1][i4+1]

                    matrix = np.zeros(shape = (5,5))

                    for vector_1 in range(12):

                        for d in range(5):

                            matrix[d][0] = vector[d][vector_1]

                        for vector_2 in range(vector_1+1,13):

                            for d in range(5):

                                matrix[d][1] = vector[d][vector_2]

                            for vector_3 in range(vector_2+1,14):

                                for d in range(5):

                                    matrix[d][2] = vector[d][vector_3]

                                for vector_4 in range(vector_3+1,15):

                                    for d in range(5):

                                        matrix[d][3] = vector[d][vector_4]

                                    for vector_5 in range(vector_4+1,16):

                                        for d in range(5):

                                            matrix[d][4] = vector[d][vector_5]

                                        weight = weight+abs(np.linalg.det(matrix))

                    weight = weight

                    if (weight > weight_0):

                        max_point[0] = fit_point[0][n1]
                        max_point[1] = fit_point[1][n2]
                        max_point[2] = fit_point[2][n3]
                        max_point[3] = fit_point[3][n4]

                        weight_0 = weight

    return max_point