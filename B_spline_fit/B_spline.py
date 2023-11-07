import math
import numpy as np

def B_spline_calculate(knot_vector,vector,order=3):

    len_knot_vector = len(knot_vector)

    i = position_find(knot_vector,vector)

    B_spline = np.zeros(shape = (len_knot_vector,order+1))

    B_spline[i][0] = 1

    for k in range(1,order+1):
            
        for n in range(i-k,i+1):

            if (knot_vector[n+k]-knot_vector[n] == 0 and (knot_vector[n+k+1]-knot_vector[n+1]) != 0):

                B_spline[n][k] = (knot_vector[n+k+1]-vector)/(knot_vector[n+k+1]-knot_vector[n+1])*B_spline[n+1][k-1]

            elif (knot_vector[n+k]-knot_vector[n] != 0 and (knot_vector[n+k+1]-knot_vector[n+1]) == 0):

                B_spline[n][k] = (vector-knot_vector[n])/(knot_vector[n+k]-knot_vector[n])*B_spline[n][k-1]

            elif (knot_vector[n+k]-knot_vector[n] == 0 and (knot_vector[n+k+1]-knot_vector[n+1]) == 0):

                B_spline[n][k] = 0

            else:

                B_spline[n][k] = (vector-knot_vector[n])/(knot_vector[n+k]-knot_vector[n])*B_spline[n][k-1]+(knot_vector[n+k+1]-vector)/(knot_vector[n+k+1]-knot_vector[n+1])*B_spline[n+1][k-1]

    return B_spline

def B_spline_point_calculate_2d(control_point,knot_vector,vector,order=3):

    size_control_point = control_point.shape
    len_control_point = size_control_point[1]

    i = position_find(knot_vector,vector)

    knot_vector = knot_vector_repeat(knot_vector,order)

    B_spline = B_spline_calculate(knot_vector,vector,order)

    point = np.zeros(2)

    for d in range(2):

        for m in range(order+1):

            if (i+m >= len_control_point):

                continue

            else:

                point[d] = point[d]+control_point[d][i+m]*B_spline[i+m][order]

    return point

def B_spline_point_calculate_3d(control_point,knot_vector,vector,order=3):

    size_control_point = control_point.shape

    i1 = position_find(knot_vector[0],vector[0])
    i2 = position_find(knot_vector[1],vector[1])

    knot_vector = list(knot_vector)

    knot_vector[0] = knot_vector_repeat(knot_vector[0],order)
    knot_vector[1] = knot_vector_repeat(knot_vector[1],order)

    B_spline_1 = B_spline_calculate(knot_vector[0],vector[0],order)
    B_spline_2 = B_spline_calculate(knot_vector[1],vector[1],order)

    point = np.zeros(3)

    for m1 in range(order+1):

        point_1 = np.zeros(2)

        for m2 in range(order+1):

            if (i2+m2 >= size_control_point[2]):

                continue

            else:

                point_1[0] = point_1[0]+control_point[1][i1+m1][i2+m2]*B_spline_2[i2+m2][order]
                point_1[1] = point_1[1]+control_point[2][i1+m1][i2+m2]*B_spline_2[i2+m2][order]

        if (i1+m1 >= size_control_point[1]):

            continue

        else:

            point[0] = point[0]+control_point[0][i1+m1][0]*B_spline_1[i1+m1][order]
            point[1] = point[1]+point_1[0]*B_spline_1[i1+m1][order]
            point[2] = point[2]+point_1[1]*B_spline_1[i1+m1][order]

    return point

def B_spline_point_calculate_4d(control_point,knot_vector,vector,order=3):

    size_control_point = control_point.shape

    i1 = position_find(knot_vector[0],vector[0])
    i2 = position_find(knot_vector[1],vector[1])
    i3 = position_find(knot_vector[2],vector[2])

    knot_vector = list(knot_vector)

    knot_vector[0] = knot_vector_repeat(knot_vector[0],order)
    knot_vector[1] = knot_vector_repeat(knot_vector[1],order)
    knot_vector[2] = knot_vector_repeat(knot_vector[2],order)

    B_spline_1 = B_spline_calculate(knot_vector[0],vector[0],order)
    B_spline_2 = B_spline_calculate(knot_vector[1],vector[1],order)
    B_spline_3 = B_spline_calculate(knot_vector[2],vector[2],order)

    point = np.zeros(4)

    for m1 in range(order+1):

        point_1 = np.zeros(3)

        for m2 in range(order+1):

            point_2 = np.zeros(2)

            for m3 in range(order+1):

                if (i3+m3 >= size_control_point[3]):

                    continue

                else:

                    point_2[0] = point_2[0]+control_point[2][i1+m1][i2+m2][i3+m3]*B_spline_3[i3+m3][order]
                    point_2[1] = point_2[1]+control_point[3][i1+m1][i2+m2][i3+m3]*B_spline_3[i3+m3][order]

            if (i2+m2 >= size_control_point[2]):

                continue

            else:

                point_1[0] = point_1[0]+control_point[1][i1+m1][i2+m2][0]*B_spline_2[i2+m2][order]
                point_1[1] = point_1[1]+point_2[0]*B_spline_2[i2+m2][order]
                point_1[2] = point_1[2]+point_2[1]*B_spline_2[i2+m2][order]

        if (i1+m1 >= size_control_point[1]):

            continue

        else:

            point[0] = point[0]+control_point[0][i1+m1][0][0]*B_spline_1[i1+m1][order]
            point[1] = point[1]+point_1[0]*B_spline_1[i1+m1][order]
            point[2] = point[2]+point_1[1]*B_spline_1[i1+m1][order]
            point[3] = point[3]+point_1[2]*B_spline_1[i1+m1][order]

    return point

def B_spline_point_calculate_5d(control_point,knot_vector,vector,order=3):

    size_control_point = control_point.shape

    i1 = position_find(knot_vector[0],vector[0])
    i2 = position_find(knot_vector[1],vector[1])
    i3 = position_find(knot_vector[2],vector[2])
    i4 = position_find(knot_vector[3],vector[3])

    knot_vector = list(knot_vector)

    knot_vector[0] = knot_vector_repeat(knot_vector[0],order)
    knot_vector[1] = knot_vector_repeat(knot_vector[1],order)
    knot_vector[2] = knot_vector_repeat(knot_vector[2],order)
    knot_vector[3] = knot_vector_repeat(knot_vector[3],order)

    B_spline_1 = B_spline_calculate(knot_vector[0],vector[0],order)
    B_spline_2 = B_spline_calculate(knot_vector[1],vector[1],order)
    B_spline_3 = B_spline_calculate(knot_vector[2],vector[2],order)
    B_spline_4 = B_spline_calculate(knot_vector[3],vector[3],order)

    point = np.zeros(5)

    for m1 in range(order+1):

        point_1 = np.zeros(4)

        for m2 in range(order+1):

            point_2 = np.zeros(3)

            for m3 in range(order+1):

                point_3 = np.zeros(2)

                for m4 in range(order+1):

                    if (i4+m4 >= size_control_point[4]):

                        continue

                    else:

                        point_3[0] = point_3[0]+control_point[3][i1+m1][i2+m2][i3+m3][i4+m4]*B_spline_4[i4+m4][order]
                        point_3[1] = point_3[1]+control_point[4][i1+m1][i2+m2][i3+m3][i4+m4]*B_spline_4[i4+m4][order]

                if (i3+m3 >= size_control_point[3]):

                    continue

                else:

                    point_2[0] = point_2[0]+control_point[2][i1+m1][i2+m2][i3+m3][0]*B_spline_3[i3+m3][order]
                    point_2[1] = point_2[1]+point_3[0]*B_spline_3[i3+m3][order]
                    point_2[2] = point_2[2]+point_3[1]*B_spline_3[i3+m3][order]

            if (i2+m2 >= size_control_point[2]):

                continue

            else:

                point_1[0] = point_1[0]+control_point[1][i1+m1][i2+m2][0][0]*B_spline_2[i2+m2][order]
                point_1[1] = point_1[1]+point_2[0]*B_spline_2[i2+m2][order]
                point_1[2] = point_1[2]+point_2[1]*B_spline_2[i2+m2][order]
                point_1[3] = point_1[3]+point_2[2]*B_spline_2[i2+m2][order]

        if (i1+m1 >= size_control_point[1]):

            continue

        else:

            point[0] = point[0]+control_point[0][i1+m1][0][0][0]*B_spline_1[i1+m1][order]
            point[1] = point[1]+point_1[0]*B_spline_1[i1+m1][order]
            point[2] = point[2]+point_1[1]*B_spline_1[i1+m1][order]
            point[3] = point[3]+point_1[2]*B_spline_1[i1+m1][order]
            point[4] = point[4]+point_1[3]*B_spline_1[i1+m1][order]

    return point

def B_spline_point_derivative_calculate_2d(control_point,knot_vector,vector,derivative_order=1,order=3):

    size_control_point = control_point.shape
    len_control_point = size_control_point[1]

    knot_vector = knot_vector_repeat(knot_vector,order)

    control_point_derivative = control_point_derivative_calculate_2d(control_point,knot_vector,derivative_order,order)   
    B_spline = B_spline_calculate(knot_vector,vector,order)

    B_spline_point_derivative = [0,0]

    for d in range(2):

        for j in range(len_control_point):

            B_spline_point_derivative[d] = B_spline_point_derivative[d]+control_point_derivative[d][j][derivative_order]*B_spline[j][order-derivative_order]

    return B_spline_point_derivative

def control_point_calculate_2d(data_point,first_tangent_vector,last_tangent_vector,knot_vector,order=3):

    size_data_point = data_point.shape
    len_data_point = size_data_point[1]

    control_point = np.zeros(shape = (2,len_data_point+2))

    max_knot_vector = max(knot_vector)

    knot_vector = knot_vector_repeat(knot_vector,order)

    for d in range(2):
        
        control_point[d][0] = data_point[d][0]
        control_point[d][1] = control_point[d][0]+knot_vector[1+order]*first_tangent_vector[d]/order
        control_point[d][len_data_point+1] = data_point[d][len_data_point-1]
        control_point[d][len_data_point] = control_point[d][len_data_point+1]-(max_knot_vector-knot_vector[len_data_point+order-2])*last_tangent_vector[d]/order

    data_point_matrix = np.delete(data_point,len_data_point-1,1)
    data_point_matrix = np.delete(data_point_matrix,0,1)

    size_data_point_matrix  = data_point_matrix.shape
    len_data_point_matrix  = size_data_point_matrix [1]

    control_point_matrix = np.zeros(shape = (2,len_data_point_matrix))
    coefficiency_matrix = np.zeros(shape = (len_data_point_matrix,len_data_point_matrix))

    for m in range(len_data_point_matrix):

        B_spline = B_spline_calculate(knot_vector,knot_vector[m+order+1],order)

        for d in range(2):

            if (m == 0):

                data_point_matrix[d][m] = data_point_matrix[d][m]-B_spline[m+1][order]*control_point[d][1]

                for i in range(order-1):

                    coefficiency_matrix[m][m+i] = B_spline[m+i+2][order]

            elif (m == len_data_point_matrix-1):

                data_point_matrix[d][m] = data_point_matrix[d][m]-B_spline[m+3][order]*control_point[d][len_data_point]

                for i in range(order):

                    if (m+i-1 >= len_data_point_matrix):

                        continue

                    else:

                        coefficiency_matrix[m][m+i-1] = B_spline[m+i+1][order]

            else:

                for i in range(order):

                    if (m+i-1 >= len_data_point_matrix):

                        continue

                    else:

                        coefficiency_matrix[m][m+i-1] = B_spline[m+i+1][order]

    coefficiency_matrix_inv = np.linalg.inv(coefficiency_matrix)

    for d in range(2):
        
        control_point_matrix[d] = np.dot(coefficiency_matrix_inv,data_point_matrix[d])

        for m in range(2,len_data_point):

            control_point[d][m] = control_point_matrix[d][m-2]

    return control_point

def control_point_calculate_3d(data_point,data_point_value,knot_vector,order=3):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])

    control_point_1 = np.zeros(shape = (3,len_data_point_1+2,len_data_point_2))

    for n2 in range(len_data_point_2):

        dp = np.zeros(shape = (2,len_data_point_1))

        for n1 in range(len_data_point_1):

            dp[0][n1] = data_point[0][n1]
            dp[1][n1] = data_point_value[n1][n2]
        
        data_point_derivative = data_point_derivative_calculate(dp,knot_vector[0])

        first_tangent_vector = [data_point_derivative[0][0][1],data_point_derivative[1][0][1]]
        last_tangent_vector = [data_point_derivative[0][len_data_point_1-order+1][1],data_point_derivative[1][len_data_point_1-order+1][1]]

        cp = control_point_calculate_2d(dp,first_tangent_vector,last_tangent_vector,knot_vector[0],order)

        for n1 in range(len_data_point_1+2):

            control_point_1[0][n1][n2] = cp[0][n1]
            control_point_1[1][n1][n2] = data_point[1][n2]
            control_point_1[2][n1][n2] = cp[1][n1]

    control_point = np.zeros(shape = (3,len_data_point_1+2,len_data_point_2+2))

    for n1 in range(len_data_point_1+2):

        dp = np.zeros(shape = (2,len_data_point_2))

        for n2 in range(len_data_point_2):

            dp[0][n2] = control_point_1[1][n1][n2]
            dp[1][n2] = control_point_1[2][n1][n2]

        data_point_derivative = data_point_derivative_calculate(dp,knot_vector[1])

        first_tangent_vector = [data_point_derivative[0][0][1],data_point_derivative[1][0][1]]
        last_tangent_vector = [data_point_derivative[0][len_data_point_2-order+1][1],data_point_derivative[1][len_data_point_2-order+1][1]]

        cp = control_point_calculate_2d(dp,first_tangent_vector,last_tangent_vector,knot_vector[1],order)

        for n2 in range(len_data_point_2+2):

            control_point[0][n1][n2] = control_point_1[0][n1][0]
            control_point[1][n1][n2] = cp[0][n2]
            control_point[2][n1][n2] = cp[1][n2]

    return control_point

def control_point_calculate_4d(data_point,data_point_value,knot_vector,order=3):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])
    len_data_point_3 = len(data_point[2])

    control_point_1 = np.zeros(shape = (4,len_data_point_1+2,len_data_point_2+2,len_data_point_3))

    for n3 in range(len_data_point_3):

        dp = [data_point[0],data_point[1]]
        kt = [knot_vector[0],knot_vector[1]]

        dpv = np.zeros(shape = (len_data_point_1,len_data_point_2))

        for n1 in range(len_data_point_1):

            for n2 in range(len_data_point_2):

                dpv[n1][n2] = data_point_value[n1][n2][n3]

        cp = control_point_calculate_3d(dp,dpv,kt,order)

        for n1 in range(len_data_point_1+2):

            for n2 in range(len_data_point_2+2):

                control_point_1[0][n1][n2][n3] = cp[0][n1][n2]
                control_point_1[1][n1][n2][n3] = cp[1][n1][n2]
                control_point_1[2][n1][n2][n3] = data_point[2][n3]
                control_point_1[3][n1][n2][n3] = cp[2][n1][n2]
    
    control_point = np.zeros(shape = (4,len_data_point_1+2,len_data_point_2+2,len_data_point_3+2))

    for n1 in range(len_data_point_1+2):

        for n2 in range(len_data_point_2+2):

            dp = np.zeros(shape = (2,len_data_point_3))

            for n3 in range(len_data_point_3):

                dp[0][n3] = control_point_1[2][n1][n2][n3]
                dp[1][n3] = control_point_1[3][n1][n2][n3]

            data_point_derivative = data_point_derivative_calculate(dp,knot_vector[2])

            first_tangent_vector = [data_point_derivative[0][0][1],data_point_derivative[1][0][1]]
            last_tangent_vector = [data_point_derivative[0][len_data_point_3-order+1][1],data_point_derivative[1][len_data_point_3-order+1][1]]

            cp = control_point_calculate_2d(dp,first_tangent_vector,last_tangent_vector,knot_vector[2],order)

            for n3 in range(len_data_point_3+2):

                control_point[0][n1][n2][n3] = control_point_1[0][n1][n2][0]
                control_point[1][n1][n2][n3] = control_point_1[1][n1][n2][0]
                control_point[2][n1][n2][n3] = cp [0][n3]
                control_point[3][n1][n2][n3] = cp [1][n3]
    
    return control_point

def control_point_calculate_5d(data_point,data_point_value,knot_vector,order=3):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])
    len_data_point_3 = len(data_point[2])
    len_data_point_4 = len(data_point[3])

    control_point_1 = np.zeros(shape = (5,len_data_point_1+2,len_data_point_2+2,len_data_point_3+2,len_data_point_4))

    for n4 in range(len_data_point_4):

        dp = [data_point[0],data_point[1],data_point[2]]
        kt = [knot_vector[0],knot_vector[1],knot_vector[2]]

        dpv = np.zeros(shape = (len_data_point_1,len_data_point_2,len_data_point_3))

        for n1 in range(len_data_point_1):

            for n2 in range(len_data_point_2):

                for n3 in range(len_data_point_3):

                    dpv[n1][n2][n3] = data_point_value[n1][n2][n3][n4]

        cp = control_point_calculate_4d(dp,dpv,kt,order)

        for n1 in range(len_data_point_1+2):

            for n2 in range(len_data_point_2+2):

                for n3 in range(len_data_point_3+2):

                    control_point_1[0][n1][n2][n3][n4] = cp[0][n1][n2][n3]
                    control_point_1[1][n1][n2][n3][n4] = cp[1][n1][n2][n3]
                    control_point_1[2][n1][n2][n3][n4] = cp[2][n1][n2][n3]
                    control_point_1[3][n1][n2][n3][n4] = data_point[3][n4]
                    control_point_1[4][n1][n2][n3][n4] = cp[3][n1][n2][n3]
    
    control_point = np.zeros(shape = (5,len_data_point_1+2,len_data_point_2+2,len_data_point_3+2,len_data_point_4+2))

    for n1 in range(len_data_point_1+2):

        for n2 in range(len_data_point_2+2):

            for n3 in range(len_data_point_3+2):

                dp = np.zeros(shape = (2,len_data_point_4))

                for n4 in range(len_data_point_4):

                    dp[0][n4] = control_point_1[3][n1][n2][n3][n4]
                    dp[1][n4] = control_point_1[4][n1][n2][n3][n4]

                data_point_derivative = data_point_derivative_calculate(dp,knot_vector[2])

                first_tangent_vector = [data_point_derivative[0][0][1],data_point_derivative[1][0][1]]
                last_tangent_vector = [data_point_derivative[0][len_data_point_4-order+1][1],data_point_derivative[1][len_data_point_4-order+1][1]]

                cp = control_point_calculate_2d(dp,first_tangent_vector,last_tangent_vector,knot_vector[3],order)

                for n4 in range(len_data_point_4+2):

                    control_point[0][n1][n2][n3][n4] = control_point_1[0][n1][n2][n3][0]
                    control_point[1][n1][n2][n3][n4] = control_point_1[1][n1][n2][n3][0]
                    control_point[2][n1][n2][n3][n4] = control_point_1[2][n1][n2][n3][0]
                    control_point[3][n1][n2][n3][n4] = cp [0][n4]
                    control_point[4][n1][n2][n3][n4] = cp [1][n4]
    
    return control_point

def control_point_derivative_calculate_2d(control_point,knot_vector,derivative_order=1,order=3):

    size_control_point = control_point.shape
    len_control_point = size_control_point[1]

    control_point_derivative = np.zeros(shape = (2,len_control_point,derivative_order+1))

    knot_vector = knot_vector_repeat(knot_vector,order)

    for d in range(2):

        for l in range(derivative_order+1):

            if (l == 0):

                for j in range(len_control_point-l):

                    control_point_derivative[d][j][l] = control_point[d][j]

            else:

                for j in range(len_control_point-l):

                    control_point_derivative[d][j][l] = (order-l+1)*(control_point_derivative[d][j+1][l-1]-control_point_derivative[d][j][l-1])/(knot_vector[j+order+1]-knot_vector[j+l])

    return control_point_derivative

def data_point_derivative_calculate(data_point,knot_vector,derivative_order=1):

    size_data_point = data_point.shape
    len_data_point = size_data_point[1]

    data_point_derivative = np.zeros(shape = (2,len_data_point,derivative_order+1))

    knot_vector_derovative = knot_vector_derivative_calculate(knot_vector,derivative_order)

    for d in range(2):

        for k in range(derivative_order+1):

            if (k == 0):

                for m in range(len_data_point-k):

                    data_point_derivative[d][m][k] = data_point[d][m]

            else:

                for m in range(len_data_point-k):

                    data_point_derivative[d][m][k] = (data_point_derivative[d][m+1][k-1]-data_point_derivative[d][m][k-1])/(knot_vector_derovative[m+1][k-1]-knot_vector_derovative[m][k-1])
    
    return data_point_derivative

def find_max_2d(data_point,point_delta,order=3):

    fit_point = fit_2d(data_point,point_delta,order)

    size_data_point = data_point.shape
    len_data_point = size_data_point[1]
    size_fit_point = fit_point.shape
    len_fit_point = size_fit_point[1]

    m = 0
    vector_1 = np.zeros(2)
    vector_2 = np.zeros(2)
    weight_0 = 0

    for n in range(len_fit_point):

        if (fit_point[0][n] > data_point[0][m+1] and m+1 <= len_data_point):

            m = m+1

        for d in range(2):

            vector_1[d] = fit_point[d][n]-data_point[d][m]
            vector_2[d] = fit_point[d][n]-data_point[d][m+1]

        weight = abs(vector_1[0]*vector_2[1]-vector_1[1]*vector_2[0])
        weight = fit_point[1][n]*point_delta+weight

        if (weight > weight_0):

            max_point = fit_point[0][n]

            weight_0 = weight

    return max_point

def fit_2d(data_point,point_delta,order=3):

    size_data_point = data_point.shape
    len_data_point = size_data_point[1]

    point_number = int((max(data_point[0])-min(data_point[0]))/point_delta)

    vector_delta = point_delta/50

    knot_vector = knot_vector_calculate_2d(data_point)

    data_point_derivative = data_point_derivative_calculate(data_point,knot_vector)

    first_tangent_vector = [data_point_derivative[0][0][1],data_point_derivative[1][0][1]]
    last_tangent_vector = [data_point_derivative[0][len_data_point-order+1][1],data_point_derivative[1][len_data_point-order+1][1]]

    control_point = control_point_calculate_2d(data_point,first_tangent_vector,last_tangent_vector,knot_vector,order)

    vector_number = int(max(knot_vector)/vector_delta)
    point_vector = np.zeros(vector_number)

    for u in range(vector_number):

        point_vector[u] = u*vector_delta

    fit_point = np.zeros(shape = (2,point_number+1))

    m = 0

    for n in range(vector_number):

        B_spline_point = B_spline_point_calculate_2d(control_point,knot_vector,point_vector[n],order)

        if (m == point_number):
            
            fit_point[0][m] = data_point[0][len_data_point-1]
            fit_point[1][m] = data_point[1][len_data_point-1]

        elif (B_spline_point[0] >= m*point_delta+min(data_point[0])):
            
            fit_point[0][m] = B_spline_point[0]
            fit_point[1][m] = B_spline_point[1]

            m = m+1

    return fit_point

def fit_2d_3d(control_point,data_point,knot_vector,point_delta,order=3):

    size_data_point = data_point.shape
    len_data_point = size_data_point[1]

    point_number = int((max(data_point[0])-min(data_point[0]))/point_delta)

    vector_delta = point_delta/50

    vector_number = int(max(knot_vector)/vector_delta)

    point_vector = np.zeros(vector_number)

    for u in range(vector_number):

        point_vector[u] = u*vector_delta

    fit_point = np.zeros(shape = (2,point_number+1))

    m = 0

    for u in range(vector_number):

        B_spline_point = B_spline_point_calculate_2d(control_point,knot_vector,point_vector[u],order)

        if (m == point_number):
            
            fit_point[0][m] = data_point[0][len_data_point-1]
            fit_point[1][m] = data_point[1][len_data_point-1]

        elif (B_spline_point[0] >= m*point_delta+min(data_point[0])):
            
            fit_point[0][m] = B_spline_point[0]
            fit_point[1][m] = B_spline_point[1]

            m = m+1

    return fit_point

def fit_2d_ac(data_point,point_delta,order=3):

    size_data_point = data_point.shape
    len_data_point = size_data_point[1]

    vector_delta = point_delta/50

    knot_vector = knot_vector_calculate_2d(data_point)

    max_knot_vector = max(knot_vector)

    data_point_derivative = data_point_derivative_calculate(data_point,knot_vector)

    first_tangent_vector = [data_point_derivative[0][0][1],data_point_derivative[1][0][1]]
    last_tangent_vector = [data_point_derivative[0][len_data_point-order+1][1],data_point_derivative[1][len_data_point-order+1][1]]

    control_point = control_point_calculate_2d(data_point,first_tangent_vector,last_tangent_vector,knot_vector,order)

    B_spline_point_derivative = B_spline_point_derivative_calculate_2d(control_point,1,knot_vector,order,vector_delta)
    first_tangent_vector = [B_spline_point_derivative[0]/np.sqrt(B_spline_point_derivative[0]*B_spline_point_derivative[0]+B_spline_point_derivative[1]*B_spline_point_derivative[1]),B_spline_point_derivative[1]/np.sqrt(B_spline_point_derivative[0]*B_spline_point_derivative[0]+B_spline_point_derivative[1]*B_spline_point_derivative[1])]
    
    B_spline_point_derivative = B_spline_point_derivative_calculate_2d(control_point,1,knot_vector,order,max_knot_vector-vector_delta)
    last_tangent_vector = [B_spline_point_derivative[0]/np.sqrt(B_spline_point_derivative[0]*B_spline_point_derivative[0]+B_spline_point_derivative[1]*B_spline_point_derivative[1]),B_spline_point_derivative[1]/np.sqrt(B_spline_point_derivative[0]*B_spline_point_derivative[0]+B_spline_point_derivative[1]*B_spline_point_derivative[1])]

    knot_point_vector = knot_point_vector_calculate_2d(control_point,data_point,knot_vector,point_delta,vector_delta,order)

    knot_vector = knot_point_vector[0]
    point_vector = knot_point_vector[1]

    control_point = control_point_calculate_2d(data_point,first_tangent_vector,last_tangent_vector,knot_vector,order)

    point_number = int((max(data_point[0])-min(data_point[0]))/point_delta)
    fit_point = np.zeros(shape = (2,point_number+1))

    for n in range(point_number+1):

        B_spline_point = B_spline_point_calculate_2d(control_point,knot_vector,point_vector[n],order)

        if (n == point_number):

            for d in range(2):

                fit_point[d][n] = data_point[d][len_data_point-1]

        else:

            for d in range(2):

                fit_point[d][n] = B_spline_point[d]

    return fit_point

def fit_3d(data_point,data_point_value,point_delta,order=3):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])

    vector_delta = np.zeros(2)

    point_number_1 = int((max(data_point[0])-min(data_point[0]))/point_delta[0])
    point_number_2 = int((max(data_point[1])-min(data_point[1]))/point_delta[1])

    for d in range(2):

        vector_delta[d] = point_delta[d]/50

    knot_vector = knot_vector_calculate_3d(data_point,data_point_value)

    control_point = control_point_calculate_3d(data_point,data_point_value,knot_vector,order)

    vector_number_1 = int(max(knot_vector[0])/vector_delta[0])
    vector_number_2 = int(max(knot_vector[1])/vector_delta[1])

    point_vector_1 = np.zeros(vector_number_1)
    point_vector_2 = np.zeros(vector_number_2)

    for u1 in range(vector_number_1):

        point_vector_1[u1] = u1*vector_delta[0]

    for u2 in range(vector_number_2):

        point_vector_2[u2] = u2*vector_delta[1]

    m1 = 0
    m2 = 0

    fit_point_1 = np.zeros(point_number_1+1)
    fit_point_2 = np.zeros(point_number_2+1)
    fit_point_value = np.zeros(shape = (point_number_1+1,point_number_2+1))

    for u1 in range(vector_number_1):

        vt = [point_vector_1[u1],point_vector_2[0]]

        B_spline_point = B_spline_point_calculate_3d(control_point,knot_vector,vt,order)

        if (B_spline_point[0] >= m1*point_delta[0]+min(data_point[0])):

            fit_point_1[m1] = B_spline_point[0]

            for u2 in range(vector_number_2):

                vt = [point_vector_1[u1],point_vector_2[u2]]

                B_spline_point = B_spline_point_calculate_3d(control_point,knot_vector,vt,order)

                if (B_spline_point[1] >= m2*point_delta[1]+min(data_point[1])):

                    fit_point_2[m2] = B_spline_point[1]
                    fit_point_value[m1][m2] = B_spline_point[2]

                    m2 = m2+1

            m1 = m1+1

            m2 = 0

    fit_point_1[point_number_1] = data_point[0][len_data_point_1-1]
    fit_point_2[point_number_2] = data_point[1][len_data_point_2-1]

    cp = np.zeros(shape = (2,len_data_point_1+2))

    for m1 in range(len_data_point_1+2):

        cp[0][m1] = control_point[0][m1][len_data_point_2+1]
        cp[1][m1] = control_point[2][m1][len_data_point_2+1]

    dp = np.zeros(shape = (2,len_data_point_1))

    for m1 in range(len_data_point_1):

        dp[0][m1] = data_point[0][m1]
        dp[1][m1] = data_point_value[m1][len_data_point_2-1]

    fit_point = fit_2d_3d(cp,dp,knot_vector[0],point_delta[0],order)

    for m1 in range(point_number_1+1):

        fit_point_value[m1][point_number_2] = fit_point[1][m1]

    cp = np.zeros(shape = (2,len_data_point_2+2))

    for m2 in range(len_data_point_2+2):

        cp[0][m2] = control_point[1][len_data_point_1+1][m2]
        cp[1][m2] = control_point[2][len_data_point_1+1][m2]

    dp = np.zeros(shape = (2,len_data_point_2))

    for m2 in range(len_data_point_2):

        dp[0][m2] = data_point[1][m2]
        dp[1][m2] = data_point_value[len_data_point_1-1][m2]

    fit_point = fit_2d_3d(cp,dp,knot_vector[1],point_delta[1],order)

    for m2 in range(point_number_2+1):

        fit_point_value[point_number_1][m2] = fit_point[1][m2]

    return fit_point_1,fit_point_2,fit_point_value

def fit_3d_4d(control_point,data_point,data_point_value,knot_vector,point_delta,order=3):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])

    vector_delta = np.zeros(2)

    point_number_1 = int((max(data_point[0])-min(data_point[0]))/point_delta[0])
    point_number_2 = int((max(data_point[1])-min(data_point[1]))/point_delta[1])

    for d in range(2):

        vector_delta[d] = point_delta[d]/50

    vector_number_1 = int(max(knot_vector[0])/vector_delta[0])
    vector_number_2 = int(max(knot_vector[1])/vector_delta[1])

    point_vector_1 = np.zeros(vector_number_1)
    point_vector_2 = np.zeros(vector_number_2)

    for u1 in range(vector_number_1):

        point_vector_1[u1] = u1*vector_delta[0]

    for u2 in range(vector_number_2):

        point_vector_2[u2] = u2*vector_delta[1]

    m1 = 0
    m2 = 0

    fit_point_1 = np.zeros(point_number_1+1)
    fit_point_2 = np.zeros(point_number_2+1)
    fit_point_value = np.zeros(shape = (point_number_1+1,point_number_2+1))

    for u1 in range(vector_number_1):

        vt = [point_vector_1[u1],point_vector_2[0]]

        B_spline_point = B_spline_point_calculate_3d(control_point,knot_vector,vt,order)

        if (B_spline_point[0] >= m1*point_delta[0]+min(data_point[0])):

            fit_point_1[m1] = B_spline_point[0]

            for u2 in range(vector_number_2):

                vt = [point_vector_1[u1],point_vector_2[u2]]

                B_spline_point = B_spline_point_calculate_3d(control_point,knot_vector,vt,order)

                if (B_spline_point[1] >= m2*point_delta[1]+min(data_point[1])):

                    fit_point_2[m2] = B_spline_point[1]
                    fit_point_value[m1][m2] = B_spline_point[2]

                    m2 = m2+1

            m1 = m1+1

            m2 = 0

    fit_point_1[point_number_1] = data_point[0][len_data_point_1-1]
    fit_point_2[point_number_2] = data_point[1][len_data_point_2-1]

    cp = np.zeros(shape = (2,len_data_point_1+2))

    for m1 in range(len_data_point_1+2):

        cp[0][m1] = control_point[0][m1][len_data_point_2+1]
        cp[1][m1] = control_point[2][m1][len_data_point_2+1]

    dp = np.zeros(shape = (2,len_data_point_1))

    for m1 in range(len_data_point_1):

        dp[0][m1] = data_point[0][m1]
        dp[1][m1] = data_point_value[m1][len_data_point_2-1]

    fit_point = fit_2d_3d(cp,dp,knot_vector[0],point_delta[0],order)

    for m1 in range(point_number_1+1):

        fit_point_value[m1][point_number_2] = fit_point[1][m1]

    cp = np.zeros(shape = (2,len_data_point_2+2))

    for m2 in range(len_data_point_2+2):

        cp[0][m2] = control_point[1][len_data_point_1+1][m2]
        cp[1][m2] = control_point[2][len_data_point_1+1][m2]

    dp = np.zeros(shape = (2,len_data_point_2))

    for m2 in range(len_data_point_2):

        dp[0][m2] = data_point[1][m2]
        dp[1][m2] = data_point_value[len_data_point_1-1][m2]

    fit_point = fit_2d_3d(cp,dp,knot_vector[1],point_delta[1],order)

    for m2 in range(point_number_2+1):

        fit_point_value[point_number_1][m2] = fit_point[1][m2]

    return fit_point_1,fit_point_2,fit_point_value

def fit_4d(data_point,data_point_value,point_delta,order=3):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])
    len_data_point_3 = len(data_point[2])

    vector_delta = np.zeros(3)

    point_number_1 = int((max(data_point[0])-min(data_point[0]))/point_delta[0])
    point_number_2 = int((max(data_point[1])-min(data_point[1]))/point_delta[1])
    point_number_3 = int((max(data_point[2])-min(data_point[2]))/point_delta[2])

    for d in range(3):

        vector_delta[d] = point_delta[d]/50

    knot_vector = knot_vector_calculate_4d(data_point,data_point_value)

    control_point = control_point_calculate_4d(data_point,data_point_value,knot_vector,order)

    vector_number_1 = int(max(knot_vector[0])/vector_delta[0])
    vector_number_2 = int(max(knot_vector[1])/vector_delta[1])
    vector_number_3 = int(max(knot_vector[2])/vector_delta[2])

    point_vector_1 = np.zeros(vector_number_1)
    point_vector_2 = np.zeros(vector_number_2)
    point_vector_3 = np.zeros(vector_number_3)

    for u1 in range(vector_number_1):

        point_vector_1[u1] = u1*vector_delta[0]

    for u2 in range(vector_number_2):

        point_vector_2[u2] = u2*vector_delta[1]

    for u3 in range(vector_number_3):

        point_vector_3[u3] = u3*vector_delta[2]

    m1 = 0
    m2 = 0
    m3 = 0

    fit_point_1 = np.zeros(point_number_1+1)
    fit_point_2 = np.zeros(point_number_2+1)
    fit_point_3 = np.zeros(point_number_3+1)
    fit_point_value = np.zeros(shape = (point_number_1+1,point_number_2+1,point_number_3+1))

    for u1 in range(vector_number_1):

        vt = [point_vector_1[u1],point_vector_2[0],point_vector_3[0]]

        B_spline_point = B_spline_point_calculate_4d(control_point,knot_vector,vt,order)

        if (B_spline_point[0] >= m1*point_delta[0]+min(data_point[0])):

            fit_point_1[m1] = B_spline_point[0]

            for u2 in range(vector_number_2):

                vt = [point_vector_1[u1],point_vector_2[u2],point_vector_3[0]]

                B_spline_point = B_spline_point_calculate_4d(control_point,knot_vector,vt,order)

                if (B_spline_point[1] >= m2*point_delta[1]+min(data_point[1])):

                    fit_point_2[m2] = B_spline_point[1]
                    
                    for u3 in range(vector_number_3):

                        vt = [point_vector_1[u1],point_vector_2[u2],point_vector_3[u3]]

                        B_spline_point = B_spline_point_calculate_4d(control_point,knot_vector,vt,order)

                        if (B_spline_point[2] >= m3*point_delta[2]+min(data_point[2])):

                            fit_point_3[m3] = B_spline_point[2]
                            fit_point_value[m1][m2][m3] = B_spline_point[3]

                            m3 = m3+1

                    m2 = m2+1

                    m3 = 0

            m1 = m1+1

            m2 = 0

    fit_point_1[point_number_1] = data_point[0][len_data_point_1-1]
    fit_point_2[point_number_2] = data_point[1][len_data_point_2-1]
    fit_point_3[point_number_3] = data_point[2][len_data_point_3-1]

    cp = np.zeros(shape = (3,len_data_point_1+2,len_data_point_2+2))

    for m1 in range(len_data_point_1+2):

        for m2 in range(len_data_point_2+2):
            
            cp[0][m1][m2] = control_point[0][m1][m2][len_data_point_3+1]
            cp[1][m1][m2] = control_point[1][m1][m2][len_data_point_3+1]
            cp[2][m1][m2] = control_point[3][m1][m2][len_data_point_3+1]
    
    dp = [data_point[0],data_point[1]]
    kt = [knot_vector[0],knot_vector[1]]
    pd = [point_delta[0],point_delta[1]]

    dpv = np.zeros(shape = (len_data_point_1,len_data_point_2))

    for m1 in range(len_data_point_1):

        for m2 in range(len_data_point_2):

            dpv[m1][m2] = data_point_value[m1][m2][len_data_point_3-1]

    fit_point = fit_3d_4d(cp,dp,dpv,kt,pd,order)

    for m1 in range(point_number_1+1):

        for m2 in range(point_number_2+1):

            fit_point_value[m1][m2][point_number_3] = fit_point[2][m1][m2]

    cp = np.zeros(shape = (3,len_data_point_1+2,len_data_point_3+2))

    for m1 in range(len_data_point_1+2):

        for m3 in range(len_data_point_3+2):
            
            cp[0][m1][m3] = control_point[0][m1][len_data_point_2+1][m3]
            cp[1][m1][m3] = control_point[2][m1][len_data_point_2+1][m3]
            cp[2][m1][m3] = control_point[3][m1][len_data_point_2+1][m3]
    
    dp = [data_point[0],data_point[2]]
    kt = [knot_vector[0],knot_vector[2]]
    pd = [point_delta[0],point_delta[2]]

    dpv = np.zeros(shape = (len_data_point_1,len_data_point_3))

    for m1 in range(len_data_point_1):

        for m3 in range(len_data_point_3):

            dpv[m1][m3] = data_point_value[m1][len_data_point_2-1][m3]

    fit_point = fit_3d_4d(cp,dp,dpv,kt,pd,order)

    for m1 in range(point_number_1+1):

        for m3 in range(point_number_3+1):

            fit_point_value[m1][point_number_2][m3] = fit_point[2][m1][m3]

    cp = np.zeros(shape = (3,len_data_point_2+2,len_data_point_3+2))

    for m2 in range(len_data_point_2+2):

        for m3 in range(len_data_point_3+2):
            
            cp[0][m2][m3] = control_point[1][len_data_point_1+1][m2][m3]
            cp[1][m2][m3] = control_point[2][len_data_point_1+1][m2][m3]
            cp[2][m2][m3] = control_point[3][len_data_point_1+1][m2][m3]
    
    dp = [data_point[1],data_point[2]]
    kt = [knot_vector[1],knot_vector[2]]
    pd = [point_delta[1],point_delta[2]]

    dpv = np.zeros(shape = (len_data_point_2,len_data_point_3))

    for m2 in range(len_data_point_2):

        for m3 in range(len_data_point_3):

            dpv[m2][m3] = data_point_value[len_data_point_1-1][m2][m3]

    fit_point = fit_3d_4d(cp,dp,dpv,kt,pd,order)

    for m2 in range(point_number_2+1):

        for m3 in range(point_number_3+1):

            fit_point_value[point_number_1][m2][m3] = fit_point[2][m2][m3]

    return fit_point_1,fit_point_2,fit_point_3,fit_point_value

def fit_4d_5d(control_point,data_point,data_point_value,knot_vector,point_delta,order=3):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])
    len_data_point_3 = len(data_point[2])

    vector_delta = np.zeros(3)

    point_number_1 = int((max(data_point[0])-min(data_point[0]))/point_delta[0])
    point_number_2 = int((max(data_point[1])-min(data_point[1]))/point_delta[1])
    point_number_3 = int((max(data_point[2])-min(data_point[2]))/point_delta[2])

    for d in range(3):

        vector_delta[d] = point_delta[d]/50

    vector_number_1 = int(max(knot_vector[0])/vector_delta[0])
    vector_number_2 = int(max(knot_vector[1])/vector_delta[1])
    vector_number_3 = int(max(knot_vector[2])/vector_delta[2])

    point_vector_1 = np.zeros(vector_number_1)
    point_vector_2 = np.zeros(vector_number_2)
    point_vector_3 = np.zeros(vector_number_3)

    for u1 in range(vector_number_1):

        point_vector_1[u1] = u1*vector_delta[0]

    for u2 in range(vector_number_2):

        point_vector_2[u2] = u2*vector_delta[1]

    for u3 in range(vector_number_3):

        point_vector_3[u3] = u3*vector_delta[2]

    m1 = 0
    m2 = 0
    m3 = 0

    fit_point_1 = np.zeros(point_number_1+1)
    fit_point_2 = np.zeros(point_number_2+1)
    fit_point_3 = np.zeros(point_number_3+1)
    fit_point_value = np.zeros(shape = (point_number_1+1,point_number_2+1,point_number_3+1))

    for u1 in range(vector_number_1):

        vt = [point_vector_1[u1],point_vector_2[0],point_vector_3[0]]

        B_spline_point = B_spline_point_calculate_4d(control_point,knot_vector,vt,order)

        if (B_spline_point[0] >= m1*point_delta[0]+min(data_point[0])):

            fit_point_1[m1] = B_spline_point[0]

            for u2 in range(vector_number_2):

                vt = [point_vector_1[u1],point_vector_2[u2],point_vector_3[0]]

                B_spline_point = B_spline_point_calculate_4d(control_point,knot_vector,vt,order)

                if (B_spline_point[1] >= m2*point_delta[1]+min(data_point[1])):

                    fit_point_2[m2] = B_spline_point[1]
                    
                    for u3 in range(vector_number_3):

                        vt = [point_vector_1[u1],point_vector_2[u2],point_vector_3[u3]]

                        B_spline_point = B_spline_point_calculate_4d(control_point,knot_vector,vt,order)

                        if (B_spline_point[2] >= m3*point_delta[2]+min(data_point[2])):

                            fit_point_3[m3] = B_spline_point[2]
                            fit_point_value[m1][m2][m3] = B_spline_point[3]

                            m3 = m3+1

                    m2 = m2+1

                    m3 = 0

            m1 = m1+1

            m2 = 0

    fit_point_1[point_number_1] = data_point[0][len_data_point_1-1]
    fit_point_2[point_number_2] = data_point[1][len_data_point_2-1]
    fit_point_3[point_number_3] = data_point[2][len_data_point_3-1]

    cp = np.zeros(shape = (3,len_data_point_1+2,len_data_point_2+2))

    for m1 in range(len_data_point_1+2):

        for m2 in range(len_data_point_2+2):
            
            cp[0][m1][m2] = control_point[0][m1][m2][len_data_point_3+1]
            cp[1][m1][m2] = control_point[1][m1][m2][len_data_point_3+1]
            cp[2][m1][m2] = control_point[3][m1][m2][len_data_point_3+1]
    
    dp = [data_point[0],data_point[1]]
    kt = [knot_vector[0],knot_vector[1]]
    pd = [point_delta[0],point_delta[1]]

    dpv = np.zeros(shape = (len_data_point_1,len_data_point_2))

    for m1 in range(len_data_point_1):

        for m2 in range(len_data_point_2):

            dpv[m1][m2] = data_point_value[m1][m2][len_data_point_3-1]

    fit_point = fit_3d_4d(cp,dp,dpv,kt,pd,order)

    for m1 in range(point_number_1+1):

        for m2 in range(point_number_2+1):

            fit_point_value[m1][m2][point_number_3] = fit_point[2][m1][m2]

    cp = np.zeros(shape = (3,len_data_point_2+2,len_data_point_3+2))

    for m2 in range(len_data_point_2+2):

        for m3 in range(len_data_point_3+2):
            
            cp[0][m2][m3] = control_point[1][len_data_point_1+1][m2][m3]
            cp[1][m2][m3] = control_point[2][len_data_point_1+1][m2][m3]
            cp[2][m2][m3] = control_point[3][len_data_point_1+1][m2][m3]
    
    dp = [data_point[1],data_point[2]]
    kt = [knot_vector[1],knot_vector[2]]
    pd = [point_delta[1],point_delta[2]]

    dpv = np.zeros(shape = (len_data_point_2,len_data_point_3))

    for m2 in range(len_data_point_2):

        for m3 in range(len_data_point_3):

            dpv[m2][m3] = data_point_value[len_data_point_1-1][m2][m3]

    fit_point = fit_3d_4d(cp,dp,dpv,kt,pd,order)

    for m2 in range(point_number_2+1):

        for m3 in range(point_number_3+1):

            fit_point_value[point_number_1][m2][m3] = fit_point[2][m2][m3]

    cp = np.zeros(shape = (3,len_data_point_1+2,len_data_point_3+2))

    for m1 in range(len_data_point_1+2):

        for m3 in range(len_data_point_3+2):
            
            cp[0][m1][m3] = control_point[0][m1][len_data_point_2+1][m3]
            cp[1][m1][m3] = control_point[2][m1][len_data_point_2+1][m3]
            cp[2][m1][m3] = control_point[3][m1][len_data_point_2+1][m3]
    
    dp = [data_point[0],data_point[2]]
    kt = [knot_vector[0],knot_vector[2]]
    pd = [point_delta[0],point_delta[2]]

    dpv = np.zeros(shape = (len_data_point_1,len_data_point_3))

    for m1 in range(len_data_point_1):

        for m3 in range(len_data_point_3):

            dpv[m1][m3] = data_point_value[m1][len_data_point_2-1][m3]

    fit_point = fit_3d_4d(cp,dp,dpv,kt,pd,order)

    for m1 in range(point_number_1+1):

        for m3 in range(point_number_3+1):

            fit_point_value[m1][point_number_2][m3] = fit_point[2][m1][m3]

    return fit_point_1,fit_point_2,fit_point_3,fit_point_value

def fit_5d(data_point,data_point_value,point_delta,order=3):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])
    len_data_point_3 = len(data_point[2])
    len_data_point_4 = len(data_point[3])

    vector_delta = np.zeros(4)

    point_number_1 = int((max(data_point[0])-min(data_point[0]))/point_delta[0])
    point_number_2 = int((max(data_point[1])-min(data_point[1]))/point_delta[1])
    point_number_3 = int((max(data_point[2])-min(data_point[2]))/point_delta[2])
    point_number_4 = int((max(data_point[3])-min(data_point[3]))/point_delta[3])

    for d in range(4):

        vector_delta[d] = point_delta[d]/20

    knot_vector = knot_vector_calculate_5d(data_point,data_point_value)

    control_point = control_point_calculate_5d(data_point,data_point_value,knot_vector,order)

    vector_number_1 = int(max(knot_vector[0])/vector_delta[0])
    vector_number_2 = int(max(knot_vector[1])/vector_delta[1])
    vector_number_3 = int(max(knot_vector[2])/vector_delta[2])
    vector_number_4 = int(max(knot_vector[3])/vector_delta[3])

    point_vector_1 = np.zeros(vector_number_1)
    point_vector_2 = np.zeros(vector_number_2)
    point_vector_3 = np.zeros(vector_number_3)
    point_vector_4 = np.zeros(vector_number_4)

    for u1 in range(vector_number_1):

        point_vector_1[u1] = u1*vector_delta[0]

    for u2 in range(vector_number_2):

        point_vector_2[u2] = u2*vector_delta[1]

    for u3 in range(vector_number_3):

        point_vector_3[u3] = u3*vector_delta[2]

    for u4 in range(vector_number_4):

        point_vector_4[u4] = u4*vector_delta[3]

    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0

    fit_point_1 = np.zeros(point_number_1+1)
    fit_point_2 = np.zeros(point_number_2+1)
    fit_point_3 = np.zeros(point_number_3+1)
    fit_point_4 = np.zeros(point_number_4+1)
    fit_point_value = np.zeros(shape = (point_number_1+1,point_number_2+1,point_number_3+1,point_number_4+1))

    for u1 in range(vector_number_1):
        
        print(vector_number_1,u1)

        vt = [point_vector_1[u1],point_vector_2[0],point_vector_3[0],point_vector_4[0]]

        B_spline_point = B_spline_point_calculate_5d(control_point,knot_vector,vt,order)

        if (B_spline_point[0] >= m1*point_delta[0]+min(data_point[0]) and m1 <= point_number_1):

            fit_point_1[m1] = B_spline_point[0]

            for u2 in range(vector_number_2):

                vt = [point_vector_1[u1],point_vector_2[u2],point_vector_3[0],point_vector_4[0]]

                B_spline_point = B_spline_point_calculate_5d(control_point,knot_vector,vt,order)

                if (B_spline_point[1] >= m2*point_delta[1]+min(data_point[1]) and m2 <= point_number_2):

                    fit_point_2[m2] = B_spline_point[1]
                    
                    for u3 in range(vector_number_3):

                        vt = [point_vector_1[u1],point_vector_2[u2],point_vector_3[u3],point_vector_4[0]]

                        B_spline_point = B_spline_point_calculate_5d(control_point,knot_vector,vt,order)

                        if (B_spline_point[2] >= m3*point_delta[2]+min(data_point[2]) and m3 <= point_number_3):

                            fit_point_3[m3] = B_spline_point[2]

                            for u4 in range(vector_number_4):

                                vt = [point_vector_1[u1],point_vector_2[u2],point_vector_3[u3],point_vector_4[u4]]

                                B_spline_point = B_spline_point_calculate_5d(control_point,knot_vector,vt,order)

                                if (B_spline_point[3] >= m4*point_delta[3]+min(data_point[3]) and m4 <= point_number_4):

                                    fit_point_4[m4] = B_spline_point[3]

                                    fit_point_value[m1][m2][m3][m4] = B_spline_point[4]

                                    m4 = m4+1

                            m3 = m3+1

                            m4 = 0

                    m2 = m2+1

                    m3 = 0

            m1 = m1+1

            m2 = 0

    fit_point_1[point_number_1] = data_point[0][len_data_point_1-1]
    fit_point_2[point_number_2] = data_point[1][len_data_point_2-1]
    fit_point_3[point_number_3] = data_point[2][len_data_point_3-1]
    fit_point_4[point_number_4] = data_point[3][len_data_point_4-1]

    cp = np.zeros(shape = (4,len_data_point_1+2,len_data_point_2+2,len_data_point_3+2))

    for m1 in range(len_data_point_1+2):

        for m2 in range(len_data_point_2+2):

            for m3 in range(len_data_point_3+2):
            
                cp[0][m1][m2][m3] = control_point[0][m1][m2][m3][len_data_point_4+1]
                cp[1][m1][m2][m3] = control_point[1][m1][m2][m3][len_data_point_4+1]
                cp[2][m1][m2][m3] = control_point[2][m1][m2][m3][len_data_point_4+1]
                cp[3][m1][m2][m3] = control_point[4][m1][m2][m3][len_data_point_4+1]
    
    dp = [data_point[0],data_point[1],data_point[2]]
    kt = [knot_vector[0],knot_vector[1],knot_vector[2]]
    pd = [point_delta[0],point_delta[1],point_delta[2]]

    dpv = np.zeros(shape = (len_data_point_1,len_data_point_2,len_data_point_3))

    for m1 in range(len_data_point_1):

        for m2 in range(len_data_point_2):

            for m3 in range(len_data_point_3):

                dpv[m1][m2][m3] = data_point_value[m1][m2][m3][len_data_point_4-1]

    fit_point = fit_4d_5d(cp,dp,dpv,kt,pd,order)

    for m1 in range(point_number_1+1):

        for m2 in range(point_number_2+1):

            for m3 in range(point_number_3+1):

                fit_point_value[m1][m2][m3][point_number_4] = fit_point[3][m1][m2][m3]

    cp = np.zeros(shape = (4,len_data_point_1+2,len_data_point_2+2,len_data_point_4+2))

    for m1 in range(len_data_point_1+2):

        for m2 in range(len_data_point_2+2):

            for m4 in range(len_data_point_4+2):
            
                cp[0][m1][m2][m4] = control_point[0][m1][m2][len_data_point_3+1][m4]
                cp[1][m1][m2][m4] = control_point[1][m1][m2][len_data_point_3+1][m4]
                cp[2][m1][m2][m4] = control_point[3][m1][m2][len_data_point_3+1][m4]
                cp[3][m1][m2][m4] = control_point[4][m1][m2][len_data_point_3+1][m4]
    
    dp = [data_point[0],data_point[1],data_point[3]]
    kt = [knot_vector[0],knot_vector[1],knot_vector[3]]
    pd = [point_delta[0],point_delta[1],point_delta[3]]

    dpv = np.zeros(shape = (len_data_point_1,len_data_point_2,len_data_point_4))

    for m1 in range(len_data_point_1):

        for m2 in range(len_data_point_2):

            for m4 in range(len_data_point_4):

                dpv[m1][m2][m4] = data_point_value[m1][m2][len_data_point_3-1][m4]

    fit_point = fit_4d_5d(cp,dp,dpv,kt,pd,order)

    for m1 in range(point_number_1+1):

        for m2 in range(point_number_2+1):

            for m4 in range(point_number_4+1):

                fit_point_value[m1][m2][point_number_3][m4] = fit_point[3][m1][m2][m4]

    cp = np.zeros(shape = (4,len_data_point_1+2,len_data_point_3+2,len_data_point_4+2))

    for m1 in range(len_data_point_1+2):

        for m3 in range(len_data_point_3+2):

            for m4 in range(len_data_point_4+2):
            
                cp[0][m1][m3][m4] = control_point[0][m1][len_data_point_2+1][m3][m4]
                cp[1][m1][m3][m4] = control_point[2][m1][len_data_point_2+1][m3][m4]
                cp[2][m1][m3][m4] = control_point[3][m1][len_data_point_2+1][m3][m4]
                cp[3][m1][m3][m4] = control_point[4][m1][len_data_point_2+1][m3][m4]
    
    dp = [data_point[0],data_point[2],data_point[3]]
    kt = [knot_vector[0],knot_vector[2],knot_vector[3]]
    pd = [point_delta[0],point_delta[2],point_delta[3]]

    dpv = np.zeros(shape = (len_data_point_1,len_data_point_3,len_data_point_4))

    for m1 in range(len_data_point_1):

        for m3 in range(len_data_point_3):

            for m4 in range(len_data_point_4):

                dpv[m1][m3][m4] = data_point_value[m1][len_data_point_2-1][m3][m4]

    fit_point = fit_4d_5d(cp,dp,dpv,kt,pd,order)

    for m1 in range(point_number_1+1):

        for m3 in range(point_number_3+1):

            for m4 in range(point_number_4+1):

                fit_point_value[m1][point_number_2][m3][m4] = fit_point[3][m1][m3][m4]

    cp = np.zeros(shape = (4,len_data_point_2+2,len_data_point_3+2,len_data_point_4+2))

    for m2 in range(len_data_point_2+2):

        for m3 in range(len_data_point_3+2):

            for m4 in range(len_data_point_4+2):
            
                cp[0][m2][m3][m4] = control_point[1][len_data_point_1+1][m2][m3][m4]
                cp[1][m2][m3][m4] = control_point[2][len_data_point_1+1][m2][m3][m4]
                cp[2][m2][m3][m4] = control_point[3][len_data_point_1+1][m2][m3][m4]
                cp[3][m2][m3][m4] = control_point[4][len_data_point_1+1][m2][m3][m4]
    
    dp = [data_point[1],data_point[2],data_point[3]]
    kt = [knot_vector[1],knot_vector[2],knot_vector[3]]
    pd = [point_delta[1],point_delta[2],point_delta[3]]

    dpv = np.zeros(shape = (len_data_point_2,len_data_point_3,len_data_point_4))

    for m2 in range(len_data_point_2):

        for m3 in range(len_data_point_3):

            for m4 in range(len_data_point_4):

                dpv[m2][m3][m4] = data_point_value[len_data_point_1-1][m2][m3][m4]

    fit_point = fit_4d_5d(cp,dp,dpv,kt,pd,order)

    for m2 in range(point_number_2+1):

        for m3 in range(point_number_3+1):

            for m4 in range(point_number_4+1):

                fit_point_value[point_number_1][m2][m3][m4] = fit_point[3][m2][m3][m4]

    return fit_point_1,fit_point_2,fit_point_3,fit_point_4,fit_point_value

def knot_point_vector_calculate_2d(control_point,data_point,knot_vector,order,point_delta,vector_delta):

    size_data_point = data_point.shape
    len_data_point = size_data_point[1]

    vector_number = int(max(knot_vector)/vector_delta)
    point_number = int((max(data_point[0])-min(data_point[0]))/point_delta)

    vector_length = 0
    CDF_point = 0
    m = 0

    new_knot_vector = np.zeros(len_data_point)
    point_vector = np.zeros(point_number+1)

    for u in range(vector_number+1):

        vector = u*vector_delta

        if (CDF_point >= m*point_delta):

            point_vector[m] = vector_length

            m = m+1

        B_spline_point_derivative = B_spline_point_derivative_calculate_2d(control_point,1,knot_vector,vector,order)

        i = position_find(knot_vector,vector)

        vector_length = vector_length+vector_delta*np.sqrt(1+(B_spline_point_derivative[1]*B_spline_point_derivative[1])/(B_spline_point_derivative[0]*B_spline_point_derivative[0]))/np.sqrt(1+(data_point[1][i+1]-data_point[1][i])*(data_point[1][i+1]-data_point[1][i])/((data_point[0][i+1]-data_point[0][i])*(data_point[0][i+1]-data_point[0][i])))
        CDF_point = CDF_point+vector_delta/np.sqrt(1+(data_point[1][i+1]-data_point[1][i])*(data_point[1][i+1]-data_point[1][i])/((data_point[0][i+1]-data_point[0][i])*(data_point[0][i+1]-data_point[0][i])))

        if (vector >= knot_vector[i]):

            new_knot_vector[i+1] = vector_length

    return new_knot_vector,point_vector

def knot_vector_calculate_2d(data_point):

    size_data_point = data_point.shape
    len_data_point = size_data_point[1]

    knot_vector = np.zeros(len_data_point)

    for n in range(len_data_point-1):

        knot_vector[n+1] = knot_vector[n]+math.sqrt((data_point[0][n+1]-data_point[0][n])*(data_point[0][n+1]-data_point[0][n])+(data_point[1][n+1]-data_point[1][n])*(data_point[1][n+1]-data_point[1][n]))

    return knot_vector

def knot_vector_calculate_2d_Hartley(control_point,data_point,order=3):

    size_control_point = control_point.shape
    len_control_point = size_control_point[1]

    control_point_length = np.zeros(len_control_point-1)

    for i in range(len_control_point-1):

        control_point_length[i] = np.sqrt((control_point[0][i+1]-control_point[0][i])*(control_point[0][i+1]-control_point[0][i])+(control_point[1][i+1]-control_point[1][i])*(control_point[1][i+1]-control_point[1][i]))

    size_data_point = data_point.shape
    len_data_point = size_data_point[1] 

    knot_vector  = np.zeros(len_data_point)
    knot_vector[len_data_point-1] = 1

    delta_length = np.zeros(len_data_point-1)
    total_length = 0

    for i in range(order,len_data_point+order-1):

        for j in range(i-order,i):

            if (j >= len_control_point-1):

                continue

            else:

                delta_length[i-order] = delta_length[i-order]+control_point_length[j]

                total_length = total_length+control_point_length[j]

    for i in range(1,len_data_point-1):
        
        knot_vector[i] = knot_vector[i-1]+delta_length[i-1]/total_length

    return knot_vector

def knot_vector_calculate_3d(data_point,data_point_value):

    len_knot_vector_1 = len(data_point[0])
    len_knot_vector_2 = len(data_point[1])

    knot_vector_1 = np.zeros(len_knot_vector_1)
    knot_vector_2 = np.zeros(len_knot_vector_2)

    for n1 in range(len_knot_vector_1-1):

        knot_vector_1[n1+1] = knot_vector_1[n1]

        for n2 in range(len_knot_vector_2):

            knot_vector_1[n1+1] = knot_vector_1[n1+1]+math.sqrt((data_point[0][n1+1]-data_point[0][n1])*(data_point[0][n1+1]-data_point[0][n1])+(data_point_value[n1+1][n2]-data_point_value[n1][n2])*(data_point_value[n1+1][n2]-data_point_value[n1][n2]))/len_knot_vector_2

    for n2 in range(len_knot_vector_2-1):

        knot_vector_2[n2+1] = knot_vector_2[n2]

        for n1 in range(len_knot_vector_1):

            knot_vector_2[n2+1] = knot_vector_2[n2+1]+math.sqrt((data_point[1][n2+1]-data_point[1][n2])*(data_point[1][n2+1]-data_point[1][n2])+(data_point_value[n1][n2+1]-data_point_value[n1][n2])*(data_point_value[n1][n2+1]-data_point_value[n1][n2]))/len_knot_vector_1

    return knot_vector_1,knot_vector_2

def knot_vector_calculate_4d(data_point,data_point_value):

    len_knot_vector_1 = len(data_point[0])
    len_knot_vector_2 = len(data_point[1])
    len_knot_vector_3 = len(data_point[2])

    knot_vector_1 = np.zeros(len_knot_vector_1)
    knot_vector_2 = np.zeros(len_knot_vector_2)
    knot_vector_3 = np.zeros(len_knot_vector_3)

    for n1 in range(len_knot_vector_1-1):

        knot_vector_1[n1+1] = knot_vector_1[n1]

        for n2 in range(len_knot_vector_2):

            for n3 in range(len_knot_vector_3):

                knot_vector_1[n1+1] = knot_vector_1[n1+1]+math.sqrt((data_point[0][n1+1]-data_point[0][n1])*(data_point[0][n1+1]-data_point[0][n1])+(data_point_value[n1+1][n2][n3]-data_point_value[n1][n2][n3])*(data_point_value[n1+1][n2][n3]-data_point_value[n1][n2][n3]))/(len_knot_vector_2*len_knot_vector_3)

    for n2 in range(len_knot_vector_2-1):

        knot_vector_2[n2+1] = knot_vector_2[n2]

        for n1 in range(len_knot_vector_1):

            for n3 in range(len_knot_vector_3):

                knot_vector_2[n2+1] = knot_vector_2[n2+1]+math.sqrt((data_point[1][n2+1]-data_point[1][n2])*(data_point[1][n2+1]-data_point[1][n2])+(data_point_value[n1][n2+1][n3]-data_point_value[n1][n2][n3])*(data_point_value[n1][n2+1][n3]-data_point_value[n1][n2][n3]))/(len_knot_vector_1*len_knot_vector_3)

    for n3 in range(len_knot_vector_3-1):

        knot_vector_3[n3+1] = knot_vector_3[n3]

        for n1 in range(len_knot_vector_1):

            for n2 in range(len_knot_vector_2):

                knot_vector_3[n3+1] = knot_vector_3[n3+1]+math.sqrt((data_point[2][n3+1]-data_point[2][n3])*(data_point[2][n3+1]-data_point[2][n3])+(data_point_value[n1][n2][n3+1]-data_point_value[n1][n2][n3])*(data_point_value[n1][n2][n3+1]-data_point_value[n1][n2][n3]))/(len_knot_vector_1*len_knot_vector_2)

    return knot_vector_1,knot_vector_2,knot_vector_3

def knot_vector_calculate_5d(data_point,data_point_value):

    len_knot_vector_1 = len(data_point[0])
    len_knot_vector_2 = len(data_point[1])
    len_knot_vector_3 = len(data_point[2])
    len_knot_vector_4 = len(data_point[3])

    knot_vector_1 = np.zeros(len_knot_vector_1)
    knot_vector_2 = np.zeros(len_knot_vector_2)
    knot_vector_3 = np.zeros(len_knot_vector_3)
    knot_vector_4 = np.zeros(len_knot_vector_4)

    for n1 in range(len_knot_vector_1-1):

        knot_vector_1[n1+1] = knot_vector_1[n1]

        for n2 in range(len_knot_vector_2):

            for n3 in range(len_knot_vector_3):

                for n4 in range(len_knot_vector_4):

                    knot_vector_1[n1+1] = knot_vector_1[n1+1]+math.sqrt((data_point[0][n1+1]-data_point[0][n1])*(data_point[0][n1+1]-data_point[0][n1])+(data_point_value[n1+1][n2][n3][n4]-data_point_value[n1][n2][n3][n4])*(data_point_value[n1+1][n2][n3][n4]-data_point_value[n1][n2][n3][n4]))/(len_knot_vector_2*len_knot_vector_3*len_knot_vector_4)

    for n2 in range(len_knot_vector_2-1):

        knot_vector_2[n2+1] = knot_vector_2[n2]

        for n1 in range(len_knot_vector_1):

            for n3 in range(len_knot_vector_3):

                for n4 in range(len_knot_vector_4):

                    knot_vector_2[n2+1] = knot_vector_2[n2+1]+math.sqrt((data_point[1][n2+1]-data_point[1][n2])*(data_point[1][n2+1]-data_point[1][n2])+(data_point_value[n1][n2+1][n3][n4]-data_point_value[n1][n2][n3][n4])*(data_point_value[n1][n2+1][n3][n4]-data_point_value[n1][n2][n3][n4]))/(len_knot_vector_1*len_knot_vector_3*len_knot_vector_4)

    for n3 in range(len_knot_vector_3-1):

        knot_vector_3[n3+1] = knot_vector_3[n3]

        for n1 in range(len_knot_vector_1):

            for n2 in range(len_knot_vector_2):

                for n4 in range(len_knot_vector_4):

                    knot_vector_3[n3+1] = knot_vector_3[n3+1]+math.sqrt((data_point[2][n3+1]-data_point[2][n3])*(data_point[2][n3+1]-data_point[2][n3])+(data_point_value[n1][n2][n3+1][n4]-data_point_value[n1][n2][n3][n4])*(data_point_value[n1][n2][n3+1][n4]-data_point_value[n1][n2][n3][n4]))/(len_knot_vector_1*len_knot_vector_2*len_knot_vector_4)

    for n4 in range(len_knot_vector_4-1):

        knot_vector_4[n4+1] = knot_vector_4[n4]

        for n1 in range(len_knot_vector_1):

            for n2 in range(len_knot_vector_2):

                for n3 in range(len_knot_vector_3):

                    knot_vector_4[n4+1] = knot_vector_4[n4+1]+math.sqrt((data_point[3][n4+1]-data_point[3][n4])*(data_point[3][n4+1]-data_point[3][n4])+(data_point_value[n1][n2][n3][n4+1]-data_point_value[n1][n2][n3][n4])*(data_point_value[n1][n2][n3][n4+1]-data_point_value[n1][n2][n3][n4]))/(len_knot_vector_1*len_knot_vector_2*len_knot_vector_3)

    return knot_vector_1,knot_vector_2,knot_vector_3,knot_vector_4

def knot_vector_derivative_calculate(knot_vector,derivative_order=1):

    len_knot_vector = len(knot_vector)

    knot_vector_derivative = np.zeros(shape = (len_knot_vector,derivative_order+1))

    for k in range(derivative_order+1):

        if (k == 0):
            
            for m in range(len_knot_vector-k):

                knot_vector_derivative[m][k] = knot_vector[m]

        else:

            for m in range(len_knot_vector-k):

                knot_vector_derivative[m][k] = (knot_vector_derivative[m][k-1]+knot_vector_derivative[m+1][k-1])/2

    return knot_vector_derivative

def knot_vector_repeat(knot_vector,order=3):

    max_knot_vector = max(knot_vector)

    knot_vector = list(knot_vector)

    for k in range(order):
        
        knot_vector.insert(0,0)
        knot_vector.append(max_knot_vector)

    return knot_vector

def point_vector_calculate_2d(data_point,knot_vector,point_delta,vector_delta):

    vector_number = int(max(knot_vector)/vector_delta)
    point_number = int((max(data_point[0])-min(data_point[0]))/point_delta)

    CDF_point = 0
    m = 0
    vector_length = 0

    point_vector = np.zeros(point_number+1)

    for u in range(vector_number+1):

        vector = u*vector_delta

        if (CDF_point >= m*point_delta):

            point_vector[m] = vector_length

            m = m+1

        i = position_find(knot_vector,vector)

        vector_length = vector_length+vector_delta
        CDF_point = CDF_point+vector_delta/np.sqrt(1+(data_point[1][i+1]-data_point[1][i])*(data_point[1][i+1]-data_point[1][i])/((data_point[0][i+1]-data_point[0][i])*(data_point[0][i+1]-data_point[0][i])))

    return point_vector

def point_vector_calculate_3d(data_point,data_point_value,knot_vector,point_delta,vector_delta):

    vector_number_1 = int(max(knot_vector[0])/vector_delta[0])
    vector_number_2 = int(max(knot_vector[1])/vector_delta[1])
    point_number_1 = int((max(data_point[0])-min(data_point[0]))/point_delta[0])
    point_number_2 = int((max(data_point[1])-min(data_point[1]))/point_delta[1])

    point_vector = np.zeros(shape = (2,point_number_1,point_number_2))

    CDF_point_1 = 0
    m1 = 0
    m2 = 0
    vector_length_1 = 0

    for u1 in range(vector_number_1+1):

        vector_1 = u1*vector_delta[0]

        if (CDF_point_1 >= m1*point_delta[0]):

            point_vector[0][m1][m2] = vector_length_1

            m1 = m1+1

        i1 = position_find(knot_vector[0],vector_1)

        vector_length_1 = vector_length_1+vector_delta[0]

        CDF_point_2 = 0
        m2 = 0
        vector_length_2 = 0

        for u2 in range(vector_number_2+1):

            vector_2 = u2*vector_delta[1]

            if (CDF_point_2 >= m2*point_delta[1]):



                point_vector[1][m1][m2] = vector_length_2

                m2 = m2+1
            
            i2 = position_find(knot_vector[1],vector_2)

            vector_length_2 = vector_length_2+vector_delta[1]
            CDF_point_2 = CDF_point_2+vector_delta[1]/np.sqrt(1+(data_point_value[i1][i2+1]-data_point_value[i1][i2])*(data_point_value[i1][i2+1]-data_point_value[i1][i2])/((data_point[1][i2+1]-data_point[1][i2])*(data_point[1][i2+1]-data_point[1][i2])))

def position_find(knot_vector,vector):

    len_knot_vector = len(knot_vector)

    for n in range(len_knot_vector-1):

        if ((vector-knot_vector[n]) >= 0 and (vector-knot_vector[n+1]) < 0):

            i = n
            break

    return i

def tangent_vector_calculate(data_point,data_point_value):

    len_knot_vector = len(data_point)

    first_tangent_vector = np.zeros(2)
    last_tangent_vector = np.zeros(2)

    first_tangent_vector[0] = (data_point[1]-data_point[0])/math.sqrt((data_point[1]-data_point[0])*(data_point[1]-data_point[0])+(data_point_value[1]-data_point_value[0])*(data_point_value[1]-data_point_value[0]))
    last_tangent_vector[0] = (data_point[len_knot_vector-1]-data_point[len_knot_vector-2])/math.sqrt((data_point[len_knot_vector-1]-data_point[len_knot_vector-2])*(data_point[len_knot_vector-1]-data_point[len_knot_vector-2])+(data_point_value[len_knot_vector-1]-data_point_value[len_knot_vector-2])*(data_point_value[len_knot_vector-1]-data_point_value[len_knot_vector-2]))
    first_tangent_vector[1] = (data_point_value[1]-data_point_value[0])/math.sqrt((data_point[1]-data_point[0])*(data_point[1]-data_point[0])+(data_point_value[1]-data_point_value[0])*(data_point_value[1]-data_point_value[0]))
    last_tangent_vector[1] = (data_point_value[len_knot_vector-1]-data_point_value[len_knot_vector-2])/math.sqrt((data_point[len_knot_vector-1]-data_point[len_knot_vector-2])*(data_point[len_knot_vector-1]-data_point[len_knot_vector-2])+(data_point_value[len_knot_vector-1]-data_point_value[len_knot_vector-2])*(data_point_value[len_knot_vector-1]-data_point_value[len_knot_vector-2]))

    return first_tangent_vector,last_tangent_vector

def tangent_vector_calculate_3d(data_point,data_point_value):

    len_knot_vector_1 = len(data_point[0])
    len_knot_vector_2 = len(data_point[1])

    first_tangent_vector_1 = np.zeros(shape = (2,len_knot_vector_2))
    last_tangent_vector_1 = np.zeros(shape = (2,len_knot_vector_2))
    first_tangent_vector_2 = np.zeros(shape = (2,len_knot_vector_1))
    last_tangent_vector_2 = np.zeros(shape = (2,len_knot_vector_1))

    for n2 in range(len_knot_vector_2):

        first_tangent_vector_1[0][n2] = (data_point[0][1]-data_point[0][0])/math.sqrt((data_point[0][1]-data_point[0][0])*(data_point[0][1]-data_point[0][0])+(data_point_value[1][n2]-data_point_value[0][n2])*(data_point_value[1][n2]-data_point_value[0][n2]))
        last_tangent_vector_1[0][n2] = (data_point[0][len_knot_vector_1-1]-data_point[0][len_knot_vector_1-2])/math.sqrt((data_point[0][len_knot_vector_1-1]-data_point[0][len_knot_vector_1-2])*(data_point[0][len_knot_vector_1-1]-data_point[0][len_knot_vector_1-2])+(data_point_value[len_knot_vector_1-1][n2]-data_point_value[len_knot_vector_1-2][n2])*(data_point_value[len_knot_vector_1-1][n2]-data_point_value[len_knot_vector_1-2][n2]))
        first_tangent_vector_1[1][n2] = (data_point_value[1][n2]-data_point_value[0][n2])/math.sqrt((data_point[0][1]-data_point[0][0])*(data_point[0][1]-data_point[0][0])+(data_point_value[1][n2]-data_point_value[0][n2])*(data_point_value[1][n2]-data_point_value[0][n2]))
        last_tangent_vector_1[1][n2] = (data_point_value[len_knot_vector_1-1][n2]-data_point_value[len_knot_vector_1-2][n2])/math.sqrt((data_point[0][len_knot_vector_1-1]-data_point[0][len_knot_vector_1-2])*(data_point[0][len_knot_vector_1-1]-data_point[0][len_knot_vector_1-2])+(data_point_value[len_knot_vector_1-1][n2]-data_point_value[len_knot_vector_1-2][n2])*(data_point_value[len_knot_vector_1-1][n2]-data_point_value[len_knot_vector_1-2][n2]))

    for n1 in range(len_knot_vector_1):

        first_tangent_vector_2[0][n1] = (data_point[1][1]-data_point[1][0])/math.sqrt((data_point[1][1]-data_point[1][0])*(data_point[1][1]-data_point[1][0])+(data_point_value[n1][1]-data_point_value[n1][0])*(data_point_value[n1][1]-data_point_value[n1][0]))
        last_tangent_vector_2[0][n1] = (data_point[1][len_knot_vector_2-1]-data_point[1][len_knot_vector_2-2])/math.sqrt((data_point[1][len_knot_vector_2-1]-data_point[1][len_knot_vector_2-2])*(data_point[1][len_knot_vector_2-1]-data_point[1][len_knot_vector_2-2])+(data_point_value[n1][len_knot_vector_2-1]-data_point_value[n1][len_knot_vector_2-2])*(data_point_value[n1][len_knot_vector_2-1]-data_point_value[n1][len_knot_vector_2-2]))
        first_tangent_vector_2[1][n1] = (data_point_value[n1][1]-data_point_value[n1][0])/math.sqrt((data_point[1][1]-data_point[1][0])*(data_point[1][1]-data_point[1][0])+(data_point_value[n1][1]-data_point_value[n1][0])*(data_point_value[n1][1]-data_point_value[n1][0]))
        last_tangent_vector_2[1][n1] = (data_point_value[n1][len_knot_vector_2-1]-data_point_value[n1][len_knot_vector_2-2])/math.sqrt((data_point[1][len_knot_vector_2-1]-data_point[1][len_knot_vector_2-2])*(data_point[1][len_knot_vector_2-1]-data_point[1][len_knot_vector_2-2])+(data_point_value[n1][len_knot_vector_2-1]-data_point_value[n1][len_knot_vector_2-2])*(data_point_value[n1][len_knot_vector_2-1]-data_point_value[n1][len_knot_vector_2-2]))

    return first_tangent_vector_1,last_tangent_vector_1,first_tangent_vector_2,last_tangent_vector_2