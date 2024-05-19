import math
import numpy as np

def B_spline_calculate(knot_vector_block,vector,order):

    len_knot_vector = len(knot_vector_block)

    i = position_find(knot_vector_block,vector)

    B_spline = np.zeros(shape = (order+1,len_knot_vector))

    B_spline[0][i] = 1

    for k in range(1,order+1):
        
        for n in range(i-k,i+1):

            if (knot_vector_block[n+k]-knot_vector_block[n] == 0 and (knot_vector_block[n+k+1]-knot_vector_block[n+1]) != 0):

                B_spline[k][n] = (knot_vector_block[n+k+1]-vector)/(knot_vector_block[n+k+1]-knot_vector_block[n+1])*B_spline[k-1][n+1]

            elif (knot_vector_block[n+k]-knot_vector_block[n] != 0 and (knot_vector_block[n+k+1]-knot_vector_block[n+1]) == 0):

                B_spline[k][n] = (vector-knot_vector_block[n])/(knot_vector_block[n+k]-knot_vector_block[n])*B_spline[k-1][n]

            elif (knot_vector_block[n+k]-knot_vector_block[n] == 0 and (knot_vector_block[n+k+1]-knot_vector_block[n+1]) == 0):

                B_spline[k][n] = 0

            else:

                B_spline[k][n] = (vector-knot_vector_block[n])/(knot_vector_block[n+k]-knot_vector_block[n])*B_spline[k-1][n]+(knot_vector_block[n+k+1]-vector)/(knot_vector_block[n+k+1]-knot_vector_block[n+1])*B_spline[k-1][n+1]

    return list(B_spline[order])

def B_spline_split(knot_vector_split,vector,spline):

    len_knot_vector_level = len(knot_vector_split)

    BS_split = np.zeros(shape = (len_knot_vector_level,len_knot_vector_level))

    BS_split[0][len_knot_vector_level-1] = spline

    i = len_knot_vector_level-2

    for k in range(1,len_knot_vector_level):

        for n in range(i-k,i+1):

            if (knot_vector_split[n+k]-knot_vector_split[n] == 0 and (knot_vector_split[n+k+1]-knot_vector_split[n+1]) != 0):

                BS_split[k][n] = (knot_vector_split[n+k+1]-vector)/(knot_vector_split[n+k+1]-knot_vector_split[n+1])*BS_split[k-1][n+1]

            elif (knot_vector_split[n+k]-knot_vector_split[n] != 0 and (knot_vector_split[n+k+1]-knot_vector_split[n+1]) == 0):

                BS_split[k][n] = (vector-knot_vector_split[n])/(knot_vector_split[n+k]-knot_vector_split[n])*BS_split[k-1][n]

            else:

                BS_split[k][n] = (vector-knot_vector_split[n])/(knot_vector_split[n+k]-knot_vector_split[n])*BS_split[k-1][n]+(knot_vector_split[n+k+1]-vector)/(knot_vector_split[n+k+1]-knot_vector_split[n+1])*BS_split[k-1][n+1]

    return BS_split

def B_spline_point_derivative_calculate(control_point,knot_vector,vector,order,derivative_order=1):

    size_control_point = control_point.shape
    len_control_point = size_control_point[1]

    knot_vector = knot_vector_repeat(knot_vector,order)

    control_point_derivative = control_point_derivative_calculate(control_point,knot_vector,order,derivative_order)   
    B_spline = B_spline_calculate(knot_vector,vector,order-derivative_order)

    B_spline_point_derivative = [0,0]

    for d in range(2):

        for j in range(len_control_point):

            B_spline_point_derivative[d] = B_spline_point_derivative[d]+control_point_derivative[d][j][derivative_order]*B_spline[j]

    return B_spline_point_derivative

def control_point_block_calculate_2d(data_point,first_tangent_vector,last_tangent_vector,knot_vector,order):

    len_data_point = len(data_point[0])

    control_point = np.zeros(shape = (2,len_data_point+2))

    max_knot_vector = max(knot_vector)

    knot_vector = knot_vector_repeat(knot_vector,order)

    for d in range(2):
        
        control_point[d][0] = data_point[d][0]
        control_point[d][1] = control_point[d][0]+knot_vector[1+order]*first_tangent_vector[d]/order/max_knot_vector
        control_point[d][len_data_point+1] = data_point[d][len_data_point-1]
        control_point[d][len_data_point] = control_point[d][len_data_point+1]-(max_knot_vector-knot_vector[len_data_point+order-2])*last_tangent_vector[d]/order/max_knot_vector

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

                data_point_matrix[d][m] = data_point_matrix[d][m]-B_spline[m+1]*control_point[d][1]

                for i in range(order-1):

                    coefficiency_matrix[m][m+i] = B_spline[m+i+2]

            elif (m == len_data_point_matrix-1):

                data_point_matrix[d][m] = data_point_matrix[d][m]-B_spline[m+3]*control_point[d][len_data_point]

                for i in range(order):

                    if (m+i-1 >= len_data_point_matrix):

                        continue

                    else:

                        coefficiency_matrix[m][m+i-1] = B_spline[m+i+1]

            else:

                for i in range(order):

                    if (m+i-1 >= len_data_point_matrix):

                        continue

                    else:

                        coefficiency_matrix[m][m+i-1] = B_spline[m+i+1]

    coefficiency_matrix_inv = np.linalg.inv(coefficiency_matrix)

    for d in range(2):
        
        control_point_matrix[d] = np.dot(coefficiency_matrix_inv,data_point_matrix[d])

        for m in range(2,len_data_point):

            control_point[d][m] = control_point_matrix[d][m-2]

    return list(control_point)

def control_point_block_calculate_3d(data_point,data_point_value,knot_vector,order):

    len_data_point_1 = len(data_point[0])
    len_data_point_2 = len(data_point[1])

    control_point_1 = np.zeros(shape = (3,len_data_point_1+2,len_data_point_2))

    for n2 in range(len_data_point_2):

        dp = np.zeros(shape = (2,len_data_point_1))

        for n1 in range(len_data_point_1):

            dp[0][n1] = data_point[0][n1]
            dp[1][n1] = data_point_value[n1][n2]
        
        tangent_vector = tangent_vector_calculate(dp[0],dp[1])

        first_tangent_vector = tangent_vector[0]
        last_tangent_vector = tangent_vector[1]

        cp = control_point_block_calculate_2d(dp,first_tangent_vector,last_tangent_vector,knot_vector[0],order[0])

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

        tangent_vector = tangent_vector_calculate(dp[0],dp[1])

        first_tangent_vector = tangent_vector[0]
        last_tangent_vector = tangent_vector[1]

        cp = control_point_block_calculate_2d(dp,first_tangent_vector,last_tangent_vector,knot_vector[1],order[1])

        for n2 in range(len_data_point_2+2):

            control_point[0][n1][n2] = control_point_1[0][n1][0]
            control_point[1][n1][n2] = cp[0][n2]
            control_point[2][n1][n2] = cp[1][n2]

    control_point_level_1 = []
    control_point_level_2 = []

    for n1 in range(len_data_point_1+2):

        control_point_level_1.append(control_point[0][n1][0])

    for n2 in range(len_data_point_2+2):

        control_point_level_2.append(control_point[1][0][n2])

    control_point_level_value = control_point[2].tolist()

    return control_point_level_1,control_point_level_2,control_point_level_value

def control_point_derivative_calculate(control_point,knot_vector,order,derivative_order=1):

    size_control_point = control_point.shape
    len_control_point = size_control_point[1]

    control_point_derivative = np.zeros(shape = (2,len_control_point,derivative_order+1))

    for d in range(2):

        for l in range(derivative_order+1):

            if (l == 0):

                for j in range(len_control_point-l):

                    control_point_derivative[d][j][l] = control_point[d][j]

            else:

                for j in range(len_control_point-l):

                    control_point_derivative[d][j][l] = (order-l+1)*(control_point_derivative[d][j+1][l-1]-control_point_derivative[d][j][l-1])/(knot_vector[j+order+1]-knot_vector[j+l])

    return control_point_derivative

def control_point_summary_2d(control_point,control_point_value,knot_vector,knot_vector_number,level):

    knot_vector_sequence = knot_vector_sequence_find(knot_vector,knot_vector_number,level)

    control_point_summary = list(control_point[0])
    control_point_value_summary = list(control_point_value[0])
    knot_vector_summary = list(knot_vector[0])

    for l in range(0,level):

        if (l == 0):

            i = position_find(knot_vector_summary,knot_vector[l+1][knot_vector_number])

            knot_vector_summary.insert(i+1,knot_vector[l+1][knot_vector_number])
            control_point_summary.insert(i+2,control_point[l][knot_vector_number])
            control_point_value_summary.insert(i+2,control_point[l][knot_vector_number])

        else:        

            i = position_find(knot_vector_summary,knot_vector[l][knot_vector_sequence[l]])

            knot_vector_summary.insert(i+1,knot_vector[l+1][knot_vector_sequence[l]])
            control_point_summary.insert(i+2,control_point[l][knot_vector_sequence[l]])
            control_point_value_summary.insert(i+2,control_point_value[l][knot_vector_sequence[l]])

    return control_point_summary,control_point_value_summary

def data_point_block_add(data_point,block,level):

    data_point_block = list(data_point[level][block])

    data_point_position = data_point_position_find(data_point,0,block,level)

    data_point_block.insert(0,data_point[data_point_position[2]][data_point_position[1]][data_point_position[0]])

    data_point_position = data_point_position_find(data_point,len(data_point[level][block])-1,block,level)

    data_point_block.append(data_point[data_point_position[5]][data_point_position[4]][data_point_position[3]])

    return data_point_block

def data_point_block_find(data_point_level,data_point_squeeze):

    b = data_point_level.index(data_point_squeeze)

    return b

def data_point_derivative_calculate(data_point_level,data_point_value_level,knot_vector_level,derivative_order=1):

    len_data_point_level = len(data_point_level)

    data_point_derivative = np.zeros(shape = (len_data_point_level,derivative_order+1))
    data_point_value_derivative = np.zeros(shape = (len_data_point_level,derivative_order+1))

    knot_vector_derovative = knot_vector_derivative_calculate(knot_vector_level,derivative_order)

    for k in range(derivative_order+1):

        if (k == 0):

            for m in range(len_data_point_level-k):

                data_point_derivative[m][k] = data_point_level[m]

        else:

            for m in range(len_data_point_level-k):

                data_point_derivative[m][k] = (data_point_derivative[m+1][k-1]-data_point_derivative[m][k-1])/(knot_vector_derovative[m+1][k-1]-knot_vector_derovative[m][k-1])

    for k in range(derivative_order+1):

        if (k == 0):

            for m in range(len_data_point_level-k):

                data_point_value_derivative[m][k] = data_point_value_level[m]

        else:

            for m in range(len_data_point_level-k):

                data_point_value_derivative[m][k] = (data_point_value_derivative[m+1][k-1]-data_point_value_derivative[m][k-1])/(knot_vector_derovative[m+1][k-1]-knot_vector_derovative[m][k-1])
    
    return data_point_derivative,data_point_value_derivative

def data_point_derivative_level_calculate_2d(control_point_level,knot_vector_level_0,vector,derivative_order=1):

    if (len(knot_vector_level_0) < 4):

        order = 2

    else:

        order = 3

    first_tangent_vector = B_spline_point_derivative_calculate(np.array(control_point_level),knot_vector_level_0,vector[0],order,derivative_order)
    last_tangent_vector = B_spline_point_derivative_calculate(np.array(control_point_level),knot_vector_level_0,vector[1],order,derivative_order)

    # a = first_tangent_vector[0]
    # b = first_tangent_vector[1]

    # first_tangent_vector[0] = a/np.sqrt(a*a+b*b)
    # first_tangent_vector[1] = b/np.sqrt(a*a+b*b)

    # a = last_tangent_vector[0]
    # b = last_tangent_vector[1]
    
    # last_tangent_vector[0] = a/np.sqrt(a*a+b*b)
    # last_tangent_vector[1] = b/np.sqrt(a*a+b*b)

    return first_tangent_vector,last_tangent_vector

def data_point_distance_calculate(data_point,data_point_value):

    data_point_distance = math.sqrt((data_point[0]-data_point[1])*(data_point[0]-data_point[1])+(data_point_value[0]-data_point_value[1])*(data_point_value[0]-data_point_value[1]))

    return data_point_distance

def data_point_position_find(data_point,number,block,level):

    min_dif = max(data_point[0][0])-min(data_point[0][0])
    max_dif = min(data_point[0][0])-max(data_point[0][0])

    for l in range(level+1):

        len_data_point_block = len(data_point[l])

        for b in range(len_data_point_block):

            len_data_point = len(data_point[l][b])

            for n in range(len_data_point):

                dif = data_point[level][block][number]-data_point[l][b][n]

                if (dif > 0 and dif < min_dif):

                    min_dif = dif

                    left_number = n
                    left_block = b
                    left_level = l

                elif (dif < 0 and dif > max_dif):

                    max_dif = dif

                    right_number = n
                    right_block = b
                    right_level = l

    return left_number,left_block,left_level,right_number,right_block,right_level

def data_point_sequence_find(data_point,number,block,level):

    data_point_sequence = []

    data_point_number = number
    data_point_block = block
    data_point_level = level

    for l in range(level):

        data_point_sequence_level = data_point[data_point_level][data_point_block]

        data_point_position = data_point_position_find(data_point,data_point_number,data_point_block,data_point_level)

        if (data_point_position[2] == data_point_level-1 or data_point_position[5] == data_point_level-1):

            if (data_point_position[2] == data_point_level-1):

                data_point_number = data_point_position[0]
                data_point_block = data_point_position[1]
                data_point_level = data_point_position[2]

            else:

                data_point_number = data_point_position[3]
                data_point_block = data_point_position[4]
                data_point_level = data_point_position[5]

        else:

            left_data_point_level = 0
            right_data_point_level = 0

            if (data_point_position[2] != 0):

                while True:

                    data_point_position = data_point_position_find(data_point,data_point_position[0],data_point_position[1],data_point_position[2])

                    if (data_point_position[2] <= data_point_level-1):

                        left_data_point_number = data_point_position[0]
                        left_data_point_block = data_point_position[1]
                        left_data_point_level = data_point_position[2]

                        break

            if (data_point_position[5] != 0):
            
                while True:

                    data_point_position = data_point_position_find(data_point,data_point_position[3],data_point_position[4],data_point_position[5])

                    if (data_point_position[5] <= data_point_level-1):

                        right_data_point_number = data_point_position[3]
                        right_data_point_block = data_point_position[4]
                        right_data_point_level = data_point_position[5]

                        break

            if (left_data_point_level > right_data_point_level):

                data_point_number = left_data_point_number
                data_point_block = left_data_point_block
                data_point_level = left_data_point_level

            else:

                data_point_number = right_data_point_number
                data_point_block = right_data_point_block
                data_point_level = right_data_point_level

        data_point_sequence.insert(0,data_point_sequence_level)

    data_point_sequence_level = data_point[0][0]

    data_point_sequence.insert(0,data_point_sequence_level)

    #     else:

            # data_point_number_sequence_level = [data_point_number]
    #         data_point_sequence_level = [data_point[data_point_level][data_point_block][data_point_number]]        

    #         data_point_position = data_point_position_find(data_point,data_point_number,data_point_level)

    #         left_number = data_point_position[0]
    #         left_block = data_point_position[1]
    #         left_level = data_point_position[2]
    #         right_number = data_point_position[3]
    #         right_block = data_point_position[4]
    #         right_level = data_point_position[5]
                
    #         if (data_point_position[2] == data_point_level or data_point_position[5] == data_point_level):

    #             if (data_point_position[2] == data_point_level):

    #                 len_data_point_block = len(data_point[data_point_level])

    #                 for b in range(len_data_point_block):

    #                     len_data_point = len(data_point[data_point_level][b])

    #                     if (data_point_position[1] == data_point_block or data_point_position[4] == data_point_block):

    #                         if (data_point_position[1] == data_point_block):

    #                             if data_point[data_point_level][data_point_block][data_point_position[0]] not in data_point_sequence_level:

    #                                 # data_point_number_sequence_level.insert(0,data_point_position[0])
    #                                 data_point_sequence_level.insert(0,data_point[data_point_level][data_point_position[1]][data_point_position[0]])

    #                             for n in range(len_data_point):

    #                                 data_point_position = data_point_position_find(data_point,data_point_position[0],data_point_position[1],data_point_position[2])

    #                                 left_number = data_point_position[0]
    #                                 left_block = data_point_position[1]
    #                                 left_level = data_point_position[2]

    #                                 if (data_point_position[1] == data_point_level):

    #                                     if data_point[data_point_level][data_point_position[1]][data_point_position[0]] not in data_point_sequence_level:

    #                                         # data_point_number_sequence_level.insert(0,data_point_position[0])
    #                                         data_point_sequence_level.insert(0,data_point[data_point_level][data_point_position[1]][data_point_position[0]])

    #                                 else:

    #                                     break

    #             if(data_point_position[5] == data_point_level):
                                
    #                 len_data_point_block = len(data_point[data_point_level])

    #                 for b in range(len_data_point_block):
                                
    #                     len_data_point = len(data_point[data_point_level][b])

    #                     if (data_point_position[1] == data_point_block or data_point_position[4] == data_point_block):
                                
    #                         if (data_point_position[1] == data_point_block):

    #                             if data_point_position[2] not in data_point_number_sequence_level:

    #                                 data_point_number_sequence_level.append(data_point_position[2])
    #                                 data_point_sequence_level.append(data_point[data_point_level][data_point_position[2]])

    #                             for n in range(len_data_point_level):

    #                                 data_point_position = data_point_position_find(data_point,data_point_position[2],data_point_position[3])

    #                                 right_number = data_point_position[2]
    #                                 right_level = data_point_position[3]

    #                                 if (data_point_position[3] == data_point_level):

    #                                     if data_point_position[2] not in data_point_number_sequence_level:

    #                                         data_point_number_sequence_level.append(data_point_position[2])
    #                                         data_point_sequence_level.append(data_point[data_point_level][data_point_position[2]])

    #                                 else:

    #                                     break

    #         if (left_level >= right_level):

    #             data_point_number = left_number
    #             data_point_level = left_level

    #         elif (left_level < right_level):

    #             data_point_number = right_number
    #             data_point_level = right_level

    #         data_point_sequence.insert(0,data_point_sequence_level)
    #         data_point_number_sequence.insert(0,data_point_number_sequence_level)
        
    return data_point_sequence

def fit_2d(data_point,data_point_value,point_delta):

    vector_delta = point_delta/10

    knot_vector = knot_vector_calculate_2d(data_point,data_point_value)

    control_point = LR_control_point_calculate_2d(data_point,data_point_value,knot_vector)

    control_point_value = control_point[1]
    control_point = control_point[0]

    point_number = int((max(data_point[0][0])-min(data_point[0][0]))/point_delta)
    vector_number = int(max(knot_vector[0][0])/vector_delta)
    point_vector = np.zeros(vector_number)

    for u in range(vector_number):

        point_vector[u] = u*vector_delta

    fit_point = np.zeros(shape = (2,point_number+1))

    m = 0

    for n in range(vector_number):

        LRHB_spline_point = LRHB_spline_point_calculate_2d(control_point,control_point_value,knot_vector,point_vector[n])

        if (m == point_number):

            fit_point[0][m] = data_point[0][0][len(data_point[0][0])-1]
            fit_point[1][m] = data_point_value[0][0][len(data_point_value[0][0])-1]

        elif (LRHB_spline_point[0] >= m*point_delta+min(data_point[0][0])):

            fit_point[0][m] = LRHB_spline_point[0]
            fit_point[1][m] = LRHB_spline_point[1]

            m = m+1

    return fit_point

def fit_2d_3d(control_point,control_point_value,data_point,data_point_value,knot_vector,point_delta):

    vector_delta = point_delta/10

    point_number = int((max(data_point[0][0])-min(data_point[0][0]))/point_delta)
    vector_number = int(max(knot_vector[0][0])/vector_delta)
    point_vector = np.zeros(vector_number)

    for u in range(vector_number):

        point_vector[u] = u*vector_delta

    fit_point = np.zeros(shape = (2,point_number+1))

    m = 0

    for n in range(vector_number):

        LRHB_spline_point = LRHB_spline_point_calculate_2d(control_point,control_point_value,knot_vector,point_vector[n])

        if (m == point_number):

            fit_point[0][m] = data_point[0][0][len(data_point[0][0])-1]
            fit_point[1][m] = data_point_value[0][0][len(data_point_value[0][0])-1]

        elif (LRHB_spline_point[0] >= m*point_delta+min(data_point[0][0])):

            fit_point[0][m] = LRHB_spline_point[0]
            fit_point[1][m] = LRHB_spline_point[1]

            m = m+1

    return fit_point

def fit_3d(data_point_1,data_point_2,data_point_value,point_delta):

    vector_delta_1 = point_delta[0]/10
    vector_delta_2 = point_delta[1]/10

    knot_vector = knot_vector_calculate_3d(data_point_1,data_point_2,data_point_value)

    knot_vector_1 = knot_vector[0]
    knot_vector_2 = knot_vector[1]

    control_point = LR_control_point_calculate_3d(data_point_1,data_point_2,data_point_value,knot_vector_1,knot_vector_2)

    control_point_1 = control_point[0]
    control_point_2 = control_point[1]
    control_point_value = control_point[2]

    point_number_1 = int((max(data_point_1[0][0])-min(data_point_1[0][0]))/point_delta[0])
    point_number_2 = int((max(data_point_2[0][0])-min(data_point_2[0][0]))/point_delta[1])
    vector_number_1 = int(max(knot_vector_1[0][0])/vector_delta_1)
    vector_number_2 = int(max(knot_vector_2[0][0])/vector_delta_2)
    point_vector_1 = np.zeros(vector_number_1)
    point_vector_2 = np.zeros(vector_number_2)

    for n1 in range(vector_number_1):

        point_vector_1[n1] = n1*vector_delta_1

    for n2 in range(vector_number_2):

        point_vector_2[n2] = n2*vector_delta_2

    fit_point_1 = list(np.zeros(point_number_1+1))
    fit_point_2 = list(np.zeros(point_number_2+1))
    fit_point_value = np.zeros(shape = (point_number_1+1,point_number_2+1))
    fit_point_value = fit_point_value.tolist()

    m1 = 0
    m2 = 0

    for n1 in range(vector_number_1):

        vt = [point_vector_1[n1],point_vector_2[0]]

        LRHB_spline_point = LRHB_spline_point_calculate_3d(control_point_1,control_point_2,control_point_value,knot_vector_1,knot_vector_2,vt)

        if (LRHB_spline_point[0] >= m1*point_delta[0]+min(data_point_1[0][0])):

            fit_point_1[m1] = LRHB_spline_point[0]

            for n2 in range(vector_number_2):

                vt = [point_vector_1[n1],point_vector_2[n2]]

                LRHB_spline_point = LRHB_spline_point_calculate_3d(control_point_1,control_point_2,control_point_value,knot_vector_1,knot_vector_2,vt)

                if (LRHB_spline_point[1] >= m2*point_delta[1]+min(data_point_2[0][0])):

                    fit_point_2[m2] = LRHB_spline_point[1]
                    fit_point_value[m1][m2] = LRHB_spline_point[2]

                    m2 = m2+1

            m1 = m1+1

            m2 = 0

    fit_point_1[point_number_1] = max(data_point_1[0][0])
    fit_point_2[point_number_2] = max(data_point_2[0][0])

    len_data_point_1 = len(data_point_1[0][0])
    len_data_point_2 = len(data_point_2[0][0])

    cpv = list(np.zeros(len_data_point_1+2))
    dpv = list(np.zeros(len_data_point_1))

    for m1 in range(len_data_point_1+2):

        cpv[m1] = control_point_value[0][0][m1][len_data_point_2+1]

    for n1 in range(len_data_point_1):

        dpv[n1] = data_point_value[0][0][n1][len_data_point_2-1]

    fit_point = fit_2d_3d([[control_point_1[0][0]]],[[cpv]],[[data_point_1[0][0]]],[[dpv]],[[knot_vector_1[0][0]]],point_delta[0])

    for m1 in range(point_number_1+1):

        fit_point_value[m1][point_number_2] = fit_point[1][m1]

    cpv = list(np.zeros(len_data_point_2+2))
    dpv = list(np.zeros(len_data_point_2))

    for m2 in range(len_data_point_2+2):

        cpv[m2] = control_point_value[0][0][len_data_point_1+1][m2]

    for n2 in range(len_data_point_2):

        dpv[n2] = data_point_value[0][0][len_data_point_1-1][n2]

    fit_point = fit_2d_3d([[control_point_2[0][0]]],[[cpv]],[[data_point_2[0][0]]],[[dpv]],[[knot_vector_2[0][0]]],point_delta[1])

    for m2 in range(point_number_2+1):

        fit_point_value[point_number_1][m2] = fit_point[1][m2]

    return fit_point_1,fit_point_2,fit_point_value

def LR_control_point_calculate_2d(data_point,data_point_value,knot_vector):

    len_data_point_level = len(data_point)

    for l in range(len_data_point_level):

        if (l == 0):

            data_point_block = data_point[0][0]
            data_point_value_block = data_point_value[0][0]
            knot_vector_block = knot_vector[0][0]

            len_data_point = len(data_point_block)

            if (len_data_point < 4):

                order = 2

            else:

                order = 3

            tangent_vector = tangent_vector_calculate(data_point[0][0],data_point_value[0][0])

            first_tangent_vector = tangent_vector[0]
            last_tangent_vector = tangent_vector[1]

            control_point_level = control_point_block_calculate_2d([data_point_block,data_point_value_block],first_tangent_vector,last_tangent_vector,knot_vector_block,order)

            control_point = [[list(control_point_level[0])]]
            control_point_value = [[list(control_point_level[1])]]

        else:

            len_data_point_block = len(data_point[l])

            control_point_level = []
            control_point_value_level = []

            for b in range(len_data_point_block):
            
                data_point_block = data_point_block_add(data_point,b,l)
                data_point_value_block = list(data_point_value[l][b])

                knot_vector_sequence = knot_vector_sequence_find(knot_vector,0,b,l)

                il1 = position_find(knot_vector_sequence[l-1],knot_vector_sequence[l][0])
                b1 = knot_vector_block_find(knot_vector[l-1],list(np.squeeze(knot_vector_sequence[l-1])),l-1)

                if (l-1 >= 1):

                    il1 = il1-1

                control_point_sequence = data_point_sequence_find(control_point,1,b1,l-1)
                control_point_value_sequence = data_point_sequence_find(control_point_value,1,b1,l-1)

                first_tangent_vector = LRHB_spline_point_derivative_calculate_2d(control_point_sequence,control_point_value_sequence,knot_vector,knot_vector_sequence,knot_vector[l-1][b1][il1],l)
                last_tangent_vector = LRHB_spline_point_derivative_calculate_2d(control_point_sequence,control_point_value_sequence,knot_vector,knot_vector_sequence,knot_vector[l-1][b1][il1+1],l)

                control_point_block = LRHBS_control_point_block_calculate_2d([control_point_sequence,control_point_value_sequence],[data_point_block,data_point_value_block],first_tangent_vector,last_tangent_vector,knot_vector,knot_vector_sequence,l)

                control_point_level.append(list(control_point_block[0]))
                control_point_value_level.append(list(control_point_block[1]))

            control_point.append(list(control_point_level))
            control_point_value.append(list(control_point_value_level))

    return control_point,control_point_value

def LR_control_point_calculate_3d(data_point_1,data_point_2,data_point_value,knot_vector_1,knot_vector_2):

    len_data_point_level = len(data_point_1)

    for l in range(len_data_point_level):

        if (l == 0):

            len_data_point_1 = len(data_point_1[0][0])
            len_data_point_2 = len(data_point_2[0][0])

            order = []

            if (len_data_point_1 < 4):

                order.append(2)

            else:

                order.append(3)

            if (len_data_point_2 < 4):

                order.append(2)

            else:

                order.append(3)

            control_point_level = control_point_block_calculate_3d([data_point_1[0][0],data_point_2[0][0]],data_point_value[0][0],[knot_vector_1[0][0],knot_vector_2[0][0]],order)

            control_point_1 = [[control_point_level[0]]]
            control_point_2 = [[control_point_level[1]]]
            control_point_value = [[control_point_level[2]]]

        else:

            len_data_point_block = len(data_point_1[l])

            control_point_1_level = []
            control_point_2_level = []
            control_point_value_level = []

            for b in range(len_data_point_block):

                control_point_block = LRHBS_control_point_block_calculate_3d(control_point_1,control_point_2,control_point_value,data_point_1,data_point_2,data_point_value,knot_vector_1,knot_vector_2,b,l)

                control_point_1_level.append(list(control_point_block[0]))
                control_point_2_level.append(list(control_point_block[1]))
                control_point_value_level.append(list(control_point_block[2]))

            control_point_1.append(list(control_point_1_level))
            control_point_2.append(list(control_point_2_level))
            control_point_value.append(list(control_point_value_level))

    return control_point_1,control_point_2,control_point_value

def LRHBS_control_point_block_calculate_2d(control_point_sequence,data_point_block,first_tangent_vector,last_tangent_vector,knot_vector,knot_vector_sequence,level):

    len_data_point = len(data_point_block[0])

    if (len_data_point < 4):

        order = 2

    else:

        order = 3

    control_point_block = np.zeros(shape = (2,len_data_point+2))

    knot_vector_block = knot_vector_repeat(knot_vector_sequence[level],order)

    data_point_matrix = np.delete(data_point_block,len_data_point-1,1)
    data_point_matrix = np.delete(data_point_matrix,0,1)

    size_data_point_matrix  = data_point_matrix.shape
    len_data_point_matrix  = size_data_point_matrix [1]

    control_point_matrix = np.zeros(shape = (2,len_data_point_matrix))
    coefficiency_matrix = np.zeros(shape = (len_data_point_matrix,len_data_point_matrix))

    for d in range(2):
        
        control_point_block[d][0] = data_point_block[d][0]

        LRHB_spline = LRHB_spline_calculate(knot_vector,knot_vector_block[order])

        for l in range(level):

            il = position_find(knot_vector_sequence[l],knot_vector_block[order])

            control_point_block[d][0] = control_point_block[d][0]-LRHB_spline[l][il]*control_point_sequence[d][l][il]-LRHB_spline[l][il+order]*control_point_sequence[d][l][il+order]
        
        control_point_block[d][0] = control_point_block[d][0]/sum(LRHB_spline[level])
        control_point_block[d][1] = control_point_block[d][0]+(knot_vector_block[1+order]-knot_vector_block[order])*first_tangent_vector[d]/order/(knot_vector_block[len_data_point+order-1]-knot_vector_block[order])
        
        control_point_block[d][len_data_point+1] = data_point_block[d][len_data_point-1]

        LRHB_spline = LRHB_spline_calculate(knot_vector,knot_vector_block[len_data_point_matrix+order+1]*(knot_vector_block[len_data_point_matrix+order]+1000)/(knot_vector_block[len_data_point_matrix+order+1]+1000))

        for l in range(level):

            il = position_find(knot_vector_sequence[l],knot_vector_block[len_data_point_matrix+order])

            control_point_block[d][len_data_point+1] = control_point_block[d][len_data_point+1]-LRHB_spline[l][il]*control_point_sequence[d][l][il]-LRHB_spline[l][il+order]*control_point_sequence[d][l][il+order]

        control_point_block[d][len_data_point+1] = control_point_block[d][len_data_point+1]/sum(LRHB_spline[level])
        control_point_block[d][len_data_point] = control_point_block[d][len_data_point+1]-(knot_vector_block[len_data_point+order-1]-knot_vector_block[len_data_point+order-2])*last_tangent_vector[d]/order/(knot_vector_block[len_data_point+order-1]-knot_vector_block[order])

    for m in range(len_data_point_matrix):

        LRHB_spline = LRHB_spline_calculate(knot_vector,knot_vector_block[m+order+1])

        for d in range(2):

            if (m == 0):

                data_point_matrix[d][m] = data_point_matrix[d][m]-LRHB_spline[level][m+1]*control_point_block[d][1]

                for l in range(level):

                    il = position_find(knot_vector_sequence[l],knot_vector_block[m+order+1])

                    data_point_matrix[d][m] = data_point_matrix[d][m]-LRHB_spline[l][il]*control_point_sequence[d][l][il]-LRHB_spline[l][il+order]*control_point_sequence[d][l][il+order]

                for i in range(order-1):

                    coefficiency_matrix[m][m+i] = LRHB_spline[level][m+i+2]

            elif (m == len_data_point_matrix-1):

                data_point_matrix[d][m] = data_point_matrix[d][m]-LRHB_spline[level][m+3]*control_point_block[d][len_data_point]

                for l in range(level):

                    il = position_find(knot_vector_sequence[l],knot_vector_block[m+order+1])

                    data_point_matrix[d][m] = data_point_matrix[d][m]-LRHB_spline[l][il]*control_point_sequence[d][l][il]-LRHB_spline[l][il+order]*control_point_sequence[d][l][il+order]

                for i in range(order):

                    if (m+i-1 >= len_data_point_matrix):

                        continue

                    else:

                        coefficiency_matrix[m][m+i-1] = LRHB_spline[level][m+i+1]

            else:

                for l in range(level):

                    il = position_find(knot_vector_sequence[l],knot_vector_block[m+order+1])

                    data_point_matrix[d][m] = data_point_matrix[d][m]-LRHB_spline[l][il]*control_point_sequence[d][l][il]-LRHB_spline[l][il+order]*control_point_sequence[d][l][il+order]

                for i in range(order):

                    if (m+i-1 >= len_data_point_matrix):

                        continue

                    else:

                        coefficiency_matrix[m][m+i-1] = LRHB_spline[level][m+i+1]

    coefficiency_matrix_inv = np.linalg.inv(coefficiency_matrix)

    for d in range(2):
        
        control_point_matrix[d] = np.dot(coefficiency_matrix_inv,data_point_matrix[d])

        for m in range(2,len_data_point):

            control_point_block[d][m] = control_point_matrix[d][m-2]

    return list(control_point_block)

def LRHBS_control_point_block_calculate_3d(control_point_1,control_point_2,control_point_value,data_point_1,data_point_2,data_point_value,knot_vector_1,knot_vector_2,block,level):

    knot_vector_1_sequence = knot_vector_sequence_find(knot_vector_1,0,block,level)
    knot_vector_2_sequence = knot_vector_sequence_find(knot_vector_2,0,block,level)

    i1 = position_find(knot_vector_1_sequence[level-1],knot_vector_1_sequence[level][0])
    i2 = position_find(knot_vector_2_sequence[level-1],knot_vector_2_sequence[level][0])
    b0 = knot_vector_block_find(knot_vector_1[level-1],list(np.squeeze(knot_vector_1_sequence[level-1])),level-1)

    data_point_1_sequence = data_point_sequence_find(data_point_1,1,b0,level-1)
    data_point_2_sequence = data_point_sequence_find(data_point_2,1,b0,level-1)
    control_point_1_sequence = data_point_sequence_find(control_point_1,1,b0,level-1)
    control_point_2_sequence = data_point_sequence_find(control_point_2,1,b0,level-1)
    control_point_value_sequence = data_point_sequence_find(control_point_value,1,b0,level-1)

    data_point_1_block = data_point_block_add(data_point_1,block,level)
    data_point_2_block = data_point_block_add(data_point_2,block,level)
    data_point_value_block = data_point_value[level][block]

    len_data_point_1 = len(data_point_1_block)
    len_data_point_2 = len(data_point_2_block)

    control_point_1 = np.zeros(shape = (3,len_data_point_1+2,len_data_point_2))

    for n2 in range(len_data_point_2):

        dp = np.zeros(shape = (2,len_data_point_1)) 
        cpvs = []

        for n1 in range(len_data_point_1):

            dp[0][n1] = data_point_1_block[n1]
            dp[1][n1] = data_point_value_block[n1][n2]

        for l in range(level):

            cpvsb = []

            for n1 in range(len(data_point_1_sequence[l])+2):

                cpvsb.append(control_point_value_sequence[l][n1][i2])

            cpvs.append(cpvsb)

        first_tangent_vector = LRHB_spline_point_derivative_calculate_2d(control_point_1_sequence,cpvs,knot_vector_1,knot_vector_1_sequence,knot_vector_1[level-1][b0][i1],level)
        last_tangent_vector = LRHB_spline_point_derivative_calculate_2d(control_point_1_sequence,cpvs,knot_vector_1,knot_vector_1_sequence,knot_vector_1[level-1][b0][i1+1],level)

        cp = LRHBS_control_point_block_calculate_2d([control_point_1_sequence,cpvs],dp,first_tangent_vector,last_tangent_vector,knot_vector_1,knot_vector_1_sequence,level)

        for n1 in range(len_data_point_1+2):

            control_point_1[0][n1][n2] = cp[0][n1]
            control_point_1[1][n1][n2] = data_point_2_block[n2]
            control_point_1[2][n1][n2] = cp[1][n1]

    control_point_block = np.zeros(shape = (3,len_data_point_1+2,len_data_point_2+2))

    for n1 in range(len_data_point_1+2):

        dp = np.zeros(shape = (2,len_data_point_2))
        cpvs = []

        for n2 in range(len_data_point_2):

            dp[0][n2] = control_point_1[1][n1][n2]
            dp[1][n2] = control_point_1[2][n1][n2]

        for l in range(level):

            cpvsb = []

            for n2 in range(len(data_point_2_sequence[l])+2):

                cpvsb.append(control_point_value_sequence[l][i1][n2])

            cpvs.append(cpvsb)

        first_tangent_vector = LRHB_spline_point_derivative_calculate_2d(control_point_2_sequence,cpvs,knot_vector_2,knot_vector_2_sequence,knot_vector_2[level-1][b0][i2],level)
        last_tangent_vector = LRHB_spline_point_derivative_calculate_2d(control_point_2_sequence,cpvs,knot_vector_2,knot_vector_2_sequence,knot_vector_2[level-1][b0][i2+1],level)

        cp = LRHBS_control_point_block_calculate_2d([control_point_2_sequence,cpvs],dp,first_tangent_vector,last_tangent_vector,knot_vector_2,knot_vector_2_sequence,level)

        for n2 in range(len_data_point_2+2):

            control_point_block[0][n1][n2] = control_point_1[0][n1][0]
            control_point_block[1][n1][n2] = cp[0][n2]
            control_point_block[2][n1][n2] = cp[1][n2]

    control_point_level_1 = []
    control_point_level_2 = []

    for n1 in range(len_data_point_1+2):

        control_point_level_1.append(control_point_block[0][n1][0])

    for n2 in range(len_data_point_2+2):

        control_point_level_2.append(control_point_block[1][0][n2])

    control_point_level_value = control_point_block[2].tolist()

    return control_point_level_1,control_point_level_2,control_point_level_value

def LRHB_spline_point_derivative_calculate_2d(control_point_sequence,control_point_value_sequence,knot_vector,knot_vector_sequence,vector,level,derivative_order=1):

    LRHB_spline_point_derivative = [0,0]

    for l in range(level):

        len_control_point_block = len(control_point_sequence[l])

        if (len_control_point_block < 6):

            order = 2

        else:

            order = 3

        knot_vector_block = knot_vector_repeat(knot_vector_sequence[l],order)

        control_point_derivative = control_point_derivative_calculate(np.array([control_point_sequence[l],control_point_value_sequence[l]]),knot_vector_block,order,derivative_order)   
        LRHB_spline = LRHB_spline_calculate(knot_vector,vector)

        for d in range(2):

            for j in range(len_control_point_block):

                LRHB_spline_point_derivative[d] = LRHB_spline_point_derivative[d]+control_point_derivative[d][j][derivative_order]*LRHB_spline[l][j]

    return LRHB_spline_point_derivative

def LRHB_spline_calculate(knot_vector,vector):

    vector_position = vector_position_find(knot_vector,vector)

    if (vector_position[2] > vector_position[5]):

        number = vector_position[0]
        block = vector_position[1]
        level = vector_position[2]

    else:

        number = vector_position[3]
        block = vector_position[4]
        level = vector_position[5]

    knot_vector_sequence = knot_vector_sequence_find(knot_vector,number,block,level)

    level = len(knot_vector_sequence)

    for l in range(level):

        if (l == 0):

            if (len(knot_vector[0][0]) < 4):

                order = 2

            else:

                order = 3

            knot_vector_sequence_level = knot_vector_sequence[0]

            knot_vector_level = knot_vector_repeat(knot_vector_sequence_level)

            spline_level = B_spline_calculate(knot_vector_level,vector,order)

            LRHB_spline = [spline_level]

        else:

            knot_vector_sequence_level = knot_vector_sequence[l]

            i = position_find(knot_vector_sequence[l-1],vector)

            if (len(knot_vector_sequence_level) < 4):

                spline_level = LRHB_spline_level_calculate_order_2(knot_vector_sequence_level,vector,LRHB_spline[l-1][i+1])

                LRHB_spline.append(spline_level)

            else:

                spline_level = LRHB_spline_level_calculate_order_3(knot_vector_sequence_level,vector,LRHB_spline[l-1][i+1],LRHB_spline[l-1][i+2])

                LRHB_spline.append(spline_level)

    return LRHB_spline

def LRHB_spline_level_calculate_order_2(knot_vector_sequence_block,vector,spline):

    order = 2

    knot_vector_sequence_block = knot_vector_repeat(knot_vector_sequence_block,order)

    LRHB_spline = B_spline_calculate(knot_vector_sequence_block,vector,order)

    for n in range(len(LRHB_spline)):

        LRHB_spline[n] = LRHB_spline[n]*spline

    return LRHB_spline

def LRHB_spline_level_calculate_order_3(knot_vector_sequence_level,vector,spline_l,spline_r):

    order = 3

    knot_vector_sequence_level = knot_vector_repeat(knot_vector_sequence_level,order)

    LRHB_spline = B_spline_calculate(knot_vector_sequence_level,vector,order)

    for n in range(len(LRHB_spline)):

        LRHB_spline[n] = LRHB_spline[n]*(spline_l+spline_r)

    return LRHB_spline

def LRHB_spline_point_calculate_2d(control_point,control_point_value,knot_vector,vector):

    vector_position = vector_position_find(knot_vector,vector)

    if (vector_position[2] > vector_position[5]):

        number = vector_position[0]
        block = vector_position[1]
        level = vector_position[2]

    else:

        number = vector_position[3]
        block = vector_position[4]
        level = vector_position[5]

    knot_vector_sequence = knot_vector_sequence_find(knot_vector,number,block,level)

    LRHB_spline = LRHB_spline_calculate(knot_vector,vector)

    point = 0
    point_value = 0

    for l in range(level+1):

        if (l == 0):

            i = position_find(knot_vector_sequence[0],vector)

            if (len(knot_vector_sequence[0]) < 4):

                order = 2

            else:

                order = 3

            for m in range(order+1):

                if (i+m >= len(control_point[0][0])):

                    continue

                else:

                    point = point+control_point[0][0][i+m]*LRHB_spline[0][i+m]
                    point_value = point_value+control_point_value[0][0][i+m]*LRHB_spline[0][i+m]

        else:

            if (len(knot_vector_sequence[l]) < 4):

                order = 2

                i1 = position_find(knot_vector_sequence[l-1],knot_vector_sequence[l][0])
                b1 = knot_vector_block_find(knot_vector[l-1],list(np.squeeze(knot_vector_sequence[l-1])),l-1)

                point = point-control_point[l-1][b1][i1+1]*LRHB_spline[l-1][i1+1]
                point_value = point-control_point_value[l-1][b1][i1+1]*LRHB_spline[l-1][i1+1]

            else:

                order = 3

                i1 = position_find(knot_vector_sequence[l-1],knot_vector_sequence[l][0])
                b1 = knot_vector_block_find(knot_vector[l-1],list(np.squeeze(knot_vector_sequence[l-1])),l-1)

                point = point-control_point[l-1][b1][i1+1]*LRHB_spline[l-1][i1+1]-control_point[l-1][b1][i1+2]*LRHB_spline[l-1][i1+2]
                point_value = point_value-control_point_value[l-1][b1][i1+1]*LRHB_spline[l-1][i1+1]-control_point_value[l-1][b1][i1+2]*LRHB_spline[l-1][i1+2]

            i = position_find(knot_vector_sequence[l],vector)
            b = knot_vector_block_find(knot_vector[l],list(np.squeeze(knot_vector_sequence[l])),l)

            for m in range(order+1):

                if (i+m >= len(control_point[l][b])):

                    continue

                else:

                    point = point+control_point[l][b][i+m]*LRHB_spline[l][i+m]
                    point_value = point_value+control_point_value[l][b][i+m]*LRHB_spline[l][i+m]

    return point,point_value

def LRHB_spline_point_calculate_3d(control_point_1,control_point_2,control_point_value,knot_vector_1,knot_vector_2,vector):

    vector_position_1 = vector_position_find(knot_vector_1,vector[0])
    vector_position_2 = vector_position_find(knot_vector_2,vector[1])

    if (vector_position_1[2] == vector_position_2[2]):

        number_1 = vector_position_1[0]
        number_2 = vector_position_2[0]
        block = vector_position_1[1]
        level = vector_position_1[2]

    elif (vector_position_1[2] == vector_position_2[5]):

        number_1 = vector_position_1[0]
        number_2 = vector_position_2[3]
        block = vector_position_1[1]
        level = vector_position_1[2]

    elif (vector_position_1[5] == vector_position_2[2]):

        number_1 = vector_position_1[5]
        number_2 = vector_position_2[2]
        block = vector_position_2[1]
        level = vector_position_2[2]

    else:

        number_1 = vector_position_1[5]
        number_2 = vector_position_2[5]
        block = vector_position_1[3]
        level = vector_position_1[4]

    knot_vector_1_sequence = knot_vector_sequence_find(knot_vector_1,number_1,block,level)
    knot_vector_2_sequence = knot_vector_sequence_find(knot_vector_2,number_2,block,level)

    LRHB_spline_1 = LRHB_spline_calculate(knot_vector_1,vector[0])
    LRHB_spline_2 = LRHB_spline_calculate(knot_vector_2,vector[1])

    point = list(np.zeros(3))
    # point_1 = []
    order = []

    for l in range(level+1):

        if (l == 0):

            i1 = position_find(knot_vector_1_sequence[0],vector[0])
            i2 = position_find(knot_vector_2_sequence[0],vector[1])

            if (len(knot_vector_1_sequence[0]) < 4):

                order.append(2)

            else:

                order.append(3)

            if (len(knot_vector_2_sequence[0]) < 4):

                order.append(2)

            else:

                order.append(3)

            # point_1_level = []

            for m1 in range(order[0]+1):

                point_1_block = list(np.zeros(2))

                for m2 in range(order[1]+1):

                    if (i2+m2 >= len(control_point_2[0][0])):

                        continue

                    else:

                        point_1_block[0] = point_1_block[0]+control_point_2[0][0][i2+m2]*LRHB_spline_2[0][i2+m2]
                        point_1_block[1] = point_1_block[1]+control_point_value[0][0][i1+m1][i2+m2]*LRHB_spline_2[0][i2+m2]

                # point_1_level.append(point_1_block)

                if (i1+m1 >= len(control_point_1[0][0])):

                    continue

                else:

                    point[0] = point[0]+control_point_1[0][0][i1+m1]*LRHB_spline_1[0][i1+m1]
                    point[1] = point[1]+point_1_block[0]*LRHB_spline_1[0][i1+m1]
                    point[2] = point[2]+point_1_block[1]*LRHB_spline_1[0][i1+m1]

        else:

            if (len(knot_vector_1_sequence[l]) < 4 and len(knot_vector_2_sequence[l]) < 4):

                order[0] = 2
                order[1] = 2

                i1 = position_find(knot_vector_1_sequence[l-1],knot_vector_1_sequence[l][0])
                i2 = position_find(knot_vector_2_sequence[l-1],knot_vector_2_sequence[l][0])
                b0 = knot_vector_block_find(knot_vector_1[l-1],list(np.squeeze(knot_vector_1_sequence[l-1])),l-1)

                point[0] = point[0]-control_point_1[l-1][b0][i1+1]*LRHB_spline_1[l-1][i1+1]
                point[1] = point[1]-control_point_2[l-1][b0][i2+1]*LRHB_spline_2[l-1][i2+1]*LRHB_spline_1[l-1][i1+1]
                point[2] = point[2]-control_point_value[l-1][b0][i1+1][i2+1]*LRHB_spline_2[l-1][i2+1]*LRHB_spline_1[l-1][i1+1]
                # point[1] = point[1]-point_1[l-1][1][0]*LRHB_spline_1[l-1][i1+1]
                # point[1] = point[1]+control_point_2[l-1][b0][i2]*LRHB_spline_2[l-1][i2]+control_point_2[l-1][b0][i2+2]*LRHB_spline_2[l-1][i2+2]
                # point[2] = point[2]-point_1[l-1][1][1]*LRHB_spline_1[l-1][i1+1]
                # point[2] = point[2]+control_point_value[l-1][b0][i1][i2]*LRHB_spline_2[l-1][i2]+control_point_value[l-1][b0][i1][i2+2]*LRHB_spline_2[l-1][i2+2]+control_point_value[l-1][b0][i1+2][i2]*LRHB_spline_2[l-1][i2]+control_point_value[l-1][b0][i1+2][i2+2]*LRHB_spline_2[l-1][i2+2]    

            elif (len(knot_vector_1_sequence[l]) < 4 and len(knot_vector_2_sequence[l]) >= 4):

                order[0] = 2
                order[1] = 3

                i1 = position_find(knot_vector_1_sequence[l-1],knot_vector_1_sequence[l][0])
                i2 = position_find(knot_vector_2_sequence[l-1],knot_vector_2_sequence[l][0])
                b0 = knot_vector_block_find(knot_vector_1[l-1],list(np.squeeze(knot_vector_1_sequence[l-1])),l-1)

                point[0] = point[0]-control_point_1[l-1][b0][i1+1]*LRHB_spline_1[l-1][i1+1]
                point[1] = point[1]-(control_point_2[l-1][b0][i2+1]*LRHB_spline_2[l-1][i2+1]+control_point_2[l-1][b0][i2+2]*LRHB_spline_2[l-1][i2+2])*LRHB_spline_1[l-1][i1+1]
                point[2] = point[2]-(control_point_value[l-1][b0][i1+1][i2+1]*LRHB_spline_2[l-1][i2+1]+control_point_value[l-1][b0][i1+1][i2+2]*LRHB_spline_2[l-1][i2+2])*LRHB_spline_1[l-1][i1+1]
                
            elif (len(knot_vector_1_sequence[l]) >= 4 and len(knot_vector_2_sequence[l]) < 4):

                order[0] = 3
                order[1] = 2

                i1 = position_find(knot_vector_1_sequence[l-1],knot_vector_1_sequence[l][0])
                i2 = position_find(knot_vector_2_sequence[l-1],knot_vector_2_sequence[l][0])
                b0 = knot_vector_block_find(knot_vector_1[l-1],list(np.squeeze(knot_vector_1_sequence[l-1])),l-1)

                point[0] = point[0]-control_point_1[l-1][b0][i1+1]*LRHB_spline_1[l-1][i1+1]-control_point_1[l-1][b0][i1+2]*LRHB_spline_1[l-1][i1+2]
                point[1] = point[1]-control_point_2[l-1][b0][i2+1]*LRHB_spline_2[l-1][i2+1]*LRHB_spline_1[l-1][i1+1]
                point[2] = point[2]-(control_point_value[l-1][b0][i1+1][i2+1]*LRHB_spline_1[l-1][i1+1]+control_point_value[l-1][b0][i1+2][i2+1]*LRHB_spline_1[l-1][i1+2])*LRHB_spline_2[l-1][i2+1]
                
            else:

                order[0] = 3
                order[1] = 3

                i1 = position_find(knot_vector_1_sequence[l-1],knot_vector_1_sequence[l][0])
                i2 = position_find(knot_vector_2_sequence[l-1],knot_vector_2_sequence[l][0])
                b0 = knot_vector_block_find(knot_vector_1[l-1],list(np.squeeze(knot_vector_1_sequence[l-1])),l-1)

                point[0] = point[0]-control_point_1[l-1][b0][i1+1]*LRHB_spline_1[l-1][i1+1]-control_point_1[l-1][b0][i1+2]*LRHB_spline_1[l-1][i1+2]
                point[1] = point[1]-(control_point_2[l-1][b0][i2+1]*LRHB_spline_2[l-1][i2+1]+control_point_2[l-1][b0][i2+2]*LRHB_spline_2[l-1][i2+2])*(LRHB_spline_1[l-1][i1+1]+LRHB_spline_1[l-1][i1+2])
                point[2] = point[2]-(control_point_value[l-1][b0][i1+1][i2+1]*LRHB_spline_2[l-1][i2+1]+control_point_value[l-1][b0][i1+1][i2+2]*LRHB_spline_2[l-1][i2+2])*LRHB_spline_1[l-1][i1+1]-(control_point_value[l-1][b0][i1+2][i2+1]*LRHB_spline_2[l-1][i2+1]+control_point_value[l-1][b0][i1+2][i2+2]*LRHB_spline_2[l-1][i2+2])*LRHB_spline_1[l-1][i1+2]
                # point[2] = point[2]-point_1[l-1][1][1]*LRHB_spline_1[l-1][i1+1]-point_1[l-1][2][1]*LRHB_spline_1[l-1][i1+2]
                # point[2] = point[2]+(control_point_value[l-1][b0][i1][i2]*LRHB_spline_2[l-1][i2]+control_point_value[l-1][b0][i1][i2+3]*LRHB_spline_2[l-1][i2+3])*LRHB_spline_1[l-1][i1]+(control_point_value[l-1][b0][i1+3][i2]*LRHB_spline_2[l-1][i2]+control_point_value[l-1][b0][i1+3][i2+3]*LRHB_spline_2[l-1][i2+3])*LRHB_spline_1[l-1][i1+3]

            i1 = position_find(knot_vector_1_sequence[l],vector[0])
            i2 = position_find(knot_vector_2_sequence[l],vector[1])
            b = knot_vector_block_find(knot_vector_1[l],list(np.squeeze(knot_vector_1_sequence[l])),l)

            # point_1_level = []

            for m1 in range(order[0]+1):

                point_1_block = list(np.zeros(2))   

                for m2 in range(order[1]+1):

                    if (i2+m2 >= len(control_point_2[l][b])):

                        continue

                    else:

                        point_1_block[0] = point_1_block[0]+control_point_2[l][b][i2+m2]*LRHB_spline_2[l][i2+m2]
                        point_1_block[1] = point_1_block[1]+control_point_value[l][b][i1+m1][i2+m2]*LRHB_spline_2[l][i2+m2]

                # point_1_level.append(point_1_block)

                if (i1+m1 >= len(control_point_1[l][b])):

                    continue

                else:

                    point[0] = point[0]+control_point_1[l][b][i1+m1]*LRHB_spline_1[l][i1+m1]
                    point[1] = point[1]+point_1_block[0]*LRHB_spline_1[l][i1+m1]
                    point[2] = point[2]+point_1_block[1]*LRHB_spline_1[l][i1+m1]

        # point_1.append(point_1_level)

    return point

def knot_vector_block_find(knot_vetor,knot_vector_squeeze,l):

    if (l == 0):

        b = data_point_block_find(knot_vetor,knot_vector_squeeze)

    else:

        knot_vector_squeeze_block = list(knot_vector_squeeze)

        knot_vector_squeeze_block.pop()
        knot_vector_squeeze_block.pop(0)

        b = knot_vetor.index(knot_vector_squeeze_block)

    return b

def knot_vector_calculate_2d(data_point,data_point_value):

    level_data_point = len(data_point)

    for l in range(level_data_point):

        if (l == 0):

            len_data_point = len(data_point[0][0])

            knot_vector = list(np.zeros(len_data_point))

            for n in range(len_data_point-1):

                knot_vector[n+1] = knot_vector[n]+data_point_distance_calculate([data_point[0][0][n+1],data_point[0][0][n]],[data_point_value[0][0][n+1],data_point_value[0][0][n]])
            
            knot_vector = [[knot_vector]]

        else:

            len_data_point_block = len(data_point[l])
            
            knot_vector_level = []

            for b in range(len_data_point_block):

                len_data_point = len(data_point[l][b])
                
                knot_vector_block = list(np.zeros(len_data_point))

                data_point_position = data_point_position_find(data_point,0,b,l)

                if (data_point_position[2] != l-2 and data_point_position[5] != l-2):

                    for n in range(len_data_point):

                        data_point_sequence = data_point_sequence_find(data_point,n,b,l)

                        il1 = position_find(data_point_sequence[l-1],data_point_sequence[l][n])
                        il = position_find(data_point_sequence[l],data_point_sequence[l][n])

                        b1 = data_point_block_find(data_point[l-1],list(np.squeeze(data_point_sequence[l-1])))

                        distance_sum = 0

                        for nd in range(len(data_point_sequence[l])+1):

                            if (nd == 0):

                                if (l-1 == 0):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point[l-1][b1][il1],data_point[l][b][nd]],[data_point_value[l-1][b1][il1],data_point_value[l][b][nd+1]])

                                else:

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point[l-1][b1][il1],data_point[l][b][nd]],[data_point_value[l-1][b1][il1+1],data_point_value[l][b][nd+1]])

                                if (nd == il):

                                    knot_distance = distance_sum
                                
                            elif (nd == len(data_point_sequence[l])):

                                if (l-1 == 0):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point[l][b][nd-1],data_point[l-1][b1][il1+1]],[data_point_value[l][b][nd],data_point_value[l-1][b1][il1+1]])

                                else:

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point[l][b][nd-1],data_point[l-1][b1][il1+1]],[data_point_value[l][b][nd],data_point_value[l-1][b1][il1+2]])

                            else:

                                distance_sum = distance_sum+data_point_distance_calculate([data_point[l][b][nd-1],data_point[l][b][nd]],[data_point_value[l][b][nd],data_point_value[l][b][nd+1]])

                                if (nd == il):

                                    knot_distance = distance_sum

                        knot_vector_block[n] = knot_vector[l-1][b1][il1]+(knot_vector[l-1][b1][il1+1]-knot_vector[l-1][b1][il1])*knot_distance/distance_sum

                else:

                    for n in range(len_data_point):

                        data_point_sequence = data_point_sequence_find(data_point,n,b,l)

                        il = position_find(data_point_sequence[l],data_point_sequence[l][n])

                        data_point_position = data_point_position_find(data_point,n,b,l)

                        ll = 0
                        rl = 0

                        if (data_point_position[2] != l and data_point_position[5] != l):

                            ln = data_point_position[0]
                            lb = data_point_position[1]
                            ll = data_point_position[2]
                            rn = data_point_position[3]
                            rb = data_point_position[4]
                            rl = data_point_position[5]

                        elif (data_point_position[2] != l and data_point_position[5] == l):

                            ln = data_point_position[0]
                            lb = data_point_position[1]
                            ll = data_point_position[2]

                            while (rl == 0):

                                data_point_position = data_point_position_find(data_point,data_point_position[3],data_point_position[4],data_point_position[5])

                                if (data_point_position[5] != l):

                                    rn = data_point_position[3]
                                    rb = data_point_position[4]
                                    rl = data_point_position[5]

                        elif (data_point_position[2] == l and data_point_position[5] != l):

                            rn = data_point_position[3]
                            rb = data_point_position[4]
                            rl = data_point_position[5]

                            while (rl == 0):

                                data_point_position = data_point_position_find(data_point,data_point_position[0],data_point_position[1],data_point_position[2])

                                if (data_point_position[2] != l):

                                    ln = data_point_position[0]
                                    lb = data_point_position[1]
                                    ll = data_point_position[2]

                        else:

                            while (rl == 0):

                                data_point_position = data_point_position_find(data_point,data_point_position[3],data_point_position[4],data_point_position[5])

                                if (data_point_position[5] != l):

                                    rn = data_point_position[3]
                                    rb = data_point_position[4]
                                    rl = data_point_position[5]

                            while (rl == 0):

                                data_point_position = data_point_position_find(data_point,data_point_position[0],data_point_position[1],data_point_position[2])

                                if (data_point_position[5] != l):

                                    ln = data_point_position[0]
                                    lb = data_point_position[1]
                                    ll = data_point_position[2]
                                    
                        distance_sum = 0

                        for nd in range(len(data_point_sequence[l])+1):

                            if (nd == 0):

                                if (ll == 0):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point[ll][lb][ln],data_point[l][b][nd]],[data_point_value[ll][lb][ln],data_point_value[l][b][nd+1]])

                                else:

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point[ll][lb][ln],data_point[l][b][nd]],[data_point_value[ll][lb][ln+1],data_point_value[l][b][nd+1]])

                                if (nd == il):

                                    knot_distance = distance_sum
                                
                            elif (nd == len(data_point_sequence[l])):

                                if (rl == 0):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point[l][b][nd-1],data_point[rl][rb][rn]],[data_point_value[l][b][nd],data_point_value[rl][rb][rn]])

                                else:

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point[l][b][nd-1],data_point[rl][rb][rn]],[data_point_value[l][b][nd],data_point_value[rl][rb][rn+1]])

                            else:

                                distance_sum = distance_sum+data_point_distance_calculate([data_point[l][b][nd-1],data_point[l][b][nd]],[data_point_value[l][b][nd],data_point_value[l][b][nd+1]])

                                if (nd == il):

                                    knot_distance = distance_sum

                        knot_vector_block[n] = knot_vector[ll][lb][ln]+(knot_vector[rl][rb][rn]-knot_vector[ll][lb][ln])*knot_distance/distance_sum

                knot_vector_level.append(knot_vector_block)

            knot_vector.append(knot_vector_level)

    return knot_vector

def knot_vector_calculate_3d(data_point_1,data_point_2,data_point_value):

    level_data_point = len(data_point_1)

    for l in range(level_data_point):

        if (l == 0):

            len_data_point_1 = len(data_point_1[0][0])
            len_data_point_2 = len(data_point_2[0][0])

            knot_vector_1_block_total = []

            for n2 in range(len_data_point_2):

                knot_vector_1_block = list(np.zeros(len_data_point_1))

                for n1 in range(len_data_point_1-1):

                    knot_vector_1_block[n1+1] = knot_vector_1_block[n1]+data_point_distance_calculate([data_point_1[0][0][n1+1],data_point_1[0][0][n1]],[data_point_value[0][0][n1+1][n2],data_point_value[0][0][n1][n2]])
                
                knot_vector_1_block_total = knot_vector_sum(knot_vector_1_block_total,knot_vector_1_block,len_data_point_2)                
            
            knot_vector_1 = [[knot_vector_1_block_total]]

        else:

            len_data_point_block = len(data_point_1[l])
            
            knot_vector_1_level = []

            for b in range(len_data_point_block):

                len_data_point_1 = len(data_point_1[l][b])
                len_data_point_2 = len(data_point_2[l][b])
                
                knot_vector_1_block_total = []

                data_point_1_position = data_point_position_find(data_point_1,0,b,l)

                if (data_point_1_position[2] >= l-2 and data_point_1_position[5] >= l-2):

                    knot_vector_1_block = list(np.zeros(len_data_point_1))

                    for n2 in range(len_data_point_2):

                        for n1 in range(len_data_point_1):

                            data_point_1_sequence = data_point_sequence_find(data_point_1,n1,b,l)

                            i1 = position_find(data_point_1_sequence[l-1],data_point_1_sequence[l][n1])
                            i0 = position_find(data_point_1_sequence[l],data_point_1_sequence[l][n1])

                            b1 = data_point_block_find(data_point_1[l-1],list(np.squeeze(data_point_1_sequence[l-1])))

                            distance_sum = 0

                            for nd in range(len(data_point_1_sequence[l])+1):

                                if (nd == 0):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_1[l-1][b1][i1],data_point_1[l][b][nd]],[data_point_value[l][b][nd][n2+1],data_point_value[l][b][nd+1][n2+1]])

                                    if (nd == i0):

                                        knot_distance = distance_sum
                                    
                                elif (nd == len(data_point_1_sequence[l])):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_1[l][b][nd-1],data_point_1[l-1][b1][i1+1]],[data_point_value[l][b][nd][n2+1],data_point_value[l][b][nd+1][n2+1]])

                                else:

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_1[l][b][nd-1],data_point_1[l][b][nd]],[data_point_value[l][b][nd][n2+1],data_point_value[l][b][nd+1][n2+1]])

                                    if (nd == i0):

                                        knot_distance = distance_sum
                            
                            knot_vector_1_block[n1] = knot_vector_1[l-1][b1][i1]+(knot_vector_1[l-1][b1][i1+1]-knot_vector_1[l-1][b1][i1])*knot_distance/distance_sum

                        knot_vector_1_block_total = knot_vector_sum(knot_vector_1_block_total,knot_vector_1_block,len_data_point_2)

                else:

                    knot_vector_1_block = list(np.zeros(len_data_point_1))
                        
                    for n2 in range(len_data_point_2):

                        for n1 in range(len_data_point_1):

                            data_point_1_sequence = data_point_sequence_find(data_point_1,n1,b,l)

                            i0 = position_find(data_point_1_sequence[l],data_point_1_sequence[l][n1])

                            data_point_1_position = data_point_position_find(data_point_1,n1,b,l)

                            ll = 0
                            rl = 0

                            if (data_point_1_position[2] != l and data_point_1_position[5] != l):

                                ln = data_point_1_position[0]
                                lb = data_point_1_position[1]
                                ll = data_point_1_position[2]
                                rn = data_point_1_position[3]
                                rb = data_point_1_position[4]
                                rl = data_point_1_position[5]

                            elif (data_point_1_position[2] != l and data_point_1_position[5] == l):

                                ln = data_point_1_position[0]
                                lb = data_point_1_position[1]
                                ll = data_point_1_position[2]

                                while (rl == 0):

                                    data_point_1_position = data_point_position_find(data_point_1,data_point_1_position[3],data_point_1_position[4],data_point_1_position[5])

                                    if (data_point_1_position[5] != l):

                                        rn = data_point_1_position[3]
                                        rb = data_point_1_position[4]
                                        rl = data_point_1_position[5]

                            elif (data_point_1_position[2] == l and data_point_1_position[5] != l):

                                rn = data_point_1_position[3]
                                rb = data_point_1_position[4]
                                rl = data_point_1_position[5]

                                while (rl == 0):

                                    data_point_1_position = data_point_position_find(data_point_1,data_point_1_position[0],data_point_1_position[1],data_point_1_position[2])

                                    if (data_point_1_position[2] != l):

                                        ln = data_point_1_position[0]
                                        lb = data_point_1_position[1]
                                        ll = data_point_1_position[2]

                            else:

                                while (rl == 0):

                                    data_point_1_position = data_point_position_find(data_point_1,data_point_1_position[3],data_point_1_position[4],data_point_1_position[5])

                                    if (data_point_1_position[5] != l):

                                        rn = data_point_1_position[3]
                                        rb = data_point_1_position[4]
                                        rl = data_point_1_position[5]

                                while (rl == 0):

                                    data_point_1_position = data_point_position_find(data_point_1,data_point_1_position[0],data_point_1_position[1],data_point_1_position[2])

                                    if (data_point_1_position[5] != l):

                                        ln = data_point_1_position[0]
                                        lb = data_point_1_position[1]
                                        ll = data_point_1_position[2]
                                        
                            distance_sum = 0

                            for nd in range(len(data_point_1_sequence[l])+1):

                                if (nd == 0):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_1[ll][lb][ln],data_point_1[l][b][nd]],[data_point_value[l][b][nd][n2+1],data_point_value[l][b][nd+1][n2+1]])

                                    if (nd == i0):

                                        knot_distance = distance_sum
                                    
                                elif (nd == len(data_point_1_sequence[l])):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_1[l][b][nd-1],data_point_1[rl][rb][rn]],[data_point_value[l][b][nd][n2+1],data_point_value[l][b][nd+1][n2+1]])

                                else:

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_1[l][b][nd-1],data_point_1[l][b][nd]],[data_point_value[l][b][nd][n2+1],data_point_value[l][b][nd+1][n2+1]])

                                    if (nd == i0):

                                        knot_distance = distance_sum

                            knot_vector_1_block[n1] = knot_vector_1[ll][lb][ln]+(knot_vector_1[rl][rb][rn]-knot_vector_1[ll][lb][ln])*knot_distance/distance_sum
                
                        knot_vector_1_block_total = knot_vector_sum(knot_vector_1_block_total,knot_vector_1_block,len_data_point_2)
                        
                knot_vector_1_level.append(knot_vector_1_block_total)

            knot_vector_1.append(knot_vector_1_level)

    for l in range(level_data_point):

        if (l == 0):

            len_data_point_1 = len(data_point_1[0][0])
            len_data_point_2 = len(data_point_2[0][0])

            knot_vector_2_block_total = []

            for n1 in range(len_data_point_1):

                knot_vector_2_block = list(np.zeros(len_data_point_2))

                for n2 in range(len_data_point_2-1):

                    knot_vector_2_block[n2+1] = knot_vector_2_block[n2]+data_point_distance_calculate([data_point_2[0][0][n2+1],data_point_2[0][0][n2]],[data_point_value[0][0][n1][n2+1],data_point_value[0][0][n1][n2]])

                knot_vector_2_block_total = knot_vector_sum(knot_vector_2_block_total,knot_vector_2_block,len_data_point_1)
            
            knot_vector_2 = [[knot_vector_2_block_total]]

        else:

            len_data_point_block = len(data_point_2[l])
            
            knot_vector_2_level = []

            for b in range(len_data_point_block):

                len_data_point_1 = len(data_point_1[l][b])
                len_data_point_2 = len(data_point_2[l][b])
                
                knot_vector_2_block_total = []

                data_point_2_position = data_point_position_find(data_point_2,0,b,l)

                if (data_point_2_position[2] != l-2 and data_point_2_position[5] != l-2):

                    knot_vector_2_block = list(np.zeros(len_data_point_2))

                    for n1 in range(len_data_point_1):

                        for n2 in range(len_data_point_2):

                            data_point_2_sequence = data_point_sequence_find(data_point_2,n2,b,l)

                            i2 = position_find(data_point_2_sequence[l-1],data_point_2_sequence[l][n2])
                            i0 = position_find(data_point_2_sequence[l],data_point_2_sequence[l][n2])

                            b2 = data_point_block_find(data_point_2[l-1],list(np.squeeze(data_point_2_sequence[l-1])))

                            distance_sum = 0

                            for nd in range(len(data_point_2_sequence[l])+1):

                                if (nd == 0):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_2[l-1][b2][i2],data_point_2[l][b][nd]],[data_point_value[l][b][n1+1][nd],data_point_value[l][b][n1+1][nd+1]])

                                    if (nd == i0):

                                        knot_distance = distance_sum
                                    
                                elif (nd == len(data_point_2_sequence[l])):

                                        distance_sum = distance_sum+data_point_distance_calculate([data_point_2[l][b][nd-1],data_point_2[l-1][b2][i2+1]],[data_point_value[l][b][n1+1][nd],data_point_value[l][b][n1+1][nd+1]])

                                else:

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_2[l][b][nd-1],data_point_2[l][b][nd]],[data_point_value[l][b][n1+1][nd],data_point_value[l][b][n1+1][nd+1]])

                                    if (nd == i0):

                                        knot_distance = distance_sum

                            knot_vector_2_block[n2] = knot_vector_2[l-1][b2][i2]+(knot_vector_2[l-1][b2][i2+1]-knot_vector_2[l-1][b2][i2])*knot_distance/distance_sum

                        knot_vector_2_block_total = knot_vector_sum(knot_vector_2_block_total,knot_vector_2_block,len_data_point_1)

                else:

                    knot_vector_2_block = list(np.zeros(len_data_point_2))
                        
                    for n1 in range(len_data_point_1):

                        for n2 in range(len_data_point_2):

                            data_point_2_sequence = data_point_sequence_find(data_point_2,n2,b,l)

                            i0 = position_find(data_point_2_sequence[l],data_point_2_sequence[l][n2])

                            data_point_2_position = data_point_position_find(data_point_2,n2,b,l)

                            ll = 0
                            rl = 0

                            if (data_point_2_position[2] != l and data_point_2_position[5] != l):

                                ln = data_point_2_position[0]
                                lb = data_point_2_position[1]
                                ll = data_point_2_position[2]
                                rn = data_point_2_position[3]
                                rb = data_point_2_position[4]
                                rl = data_point_2_position[5]

                            elif (data_point_2_position[2] != l and data_point_2_position[5] == l):

                                ln = data_point_2_position[0]
                                lb = data_point_2_position[1]
                                ll = data_point_2_position[2]

                                while (rl == 0):

                                    data_point_2_position = data_point_position_find(data_point_2,data_point_2_position[3],data_point_2_position[4],data_point_2_position[5])

                                    if (data_point_2_position[5] != l):

                                        rn = data_point_2_position[3]
                                        rb = data_point_2_position[4]
                                        rl = data_point_2_position[5]

                            elif (data_point_2_position[2] == l and data_point_2_position[5] != l):

                                rn = data_point_2_position[3]
                                rb = data_point_2_position[4]
                                rl = data_point_2_position[5]

                                while (rl == 0):

                                    data_point_2_position = data_point_position_find(data_point_2,data_point_2_position[0],data_point_2_position[1],data_point_2_position[2])

                                    if (data_point_2_position[2] != l):

                                        ln = data_point_2_position[0]
                                        lb = data_point_2_position[1]
                                        ll = data_point_2_position[2]

                            else:

                                while (rl == 0):

                                    data_point_2_position = data_point_position_find(data_point_2,data_point_2_position[3],data_point_2_position[4],data_point_2_position[5])

                                    if (data_point_2_position[5] != l):

                                        rn = data_point_2_position[3]
                                        rb = data_point_2_position[4]
                                        rl = data_point_2_position[5]

                                while (rl == 0):

                                    data_point_2_position = data_point_position_find(data_point_2,data_point_2_position[0],data_point_2_position[1],data_point_2_position[2])

                                    if (data_point_2_position[5] != l):

                                        ln = data_point_2_position[0]
                                        lb = data_point_2_position[1]
                                        ll = data_point_2_position[2]
                                        
                            distance_sum = 0

                            for nd in range(len(data_point_2_sequence[l])+1):

                                if (nd == 0):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_2[ll][lb][ln],data_point_2[l][b][nd]],[data_point_value[l][b][n1+1][nd],data_point_value[l][b][n1+1][nd+1]])

                                    if (nd == i0):

                                        knot_distance = distance_sum
                                    
                                elif (nd == len(data_point_2_sequence[l])):

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_2[l][b][nd-1],data_point_2[rl][rb][rn]],[data_point_value[l][b][n1+1][nd],data_point_value[l][b][n1+1][nd+1]])

                                else:

                                    distance_sum = distance_sum+data_point_distance_calculate([data_point_2[l][b][nd-1],data_point_2[l][b][nd]],[data_point_value[l][b][n1+1][nd],data_point_value[l][b][n1+1][nd+1]])

                                    if (nd == i0):

                                        knot_distance = distance_sum

                            knot_vector_2_block[n2] = knot_vector_2[ll][lb][ln]+(knot_vector_2[rl][rb][rn]-knot_vector_2[ll][lb][ln])*knot_distance/distance_sum
                        
                        knot_vector_2_block_total = knot_vector_sum(knot_vector_2_block_total,knot_vector_2_block,len_data_point_1)
                        
                knot_vector_2_level.append(knot_vector_2_block_total)

            knot_vector_2.append(knot_vector_2_level)

    return knot_vector_1,knot_vector_2

def knot_vector_derivative_calculate(knot_vector_level,derivative_order):

    len_knot_vector_level = len(knot_vector_level)

    knot_vector_derivative = np.zeros(shape = (len_knot_vector_level,derivative_order+1))

    for k in range(derivative_order+1):

        if (k == 0):
            
            for m in range(len_knot_vector_level-k):

                knot_vector_derivative[m][k] = knot_vector_level[m]

        else:

            for m in range(len_knot_vector_level-k):

                knot_vector_derivative[m][k] = (knot_vector_derivative[m][k-1]+knot_vector_derivative[m+1][k-1])/2

    return knot_vector_derivative

def knot_vector_sequence_find(knot_vector,number,block,level):

    data_sequence = data_point_sequence_find(knot_vector,number,block,level)

    knot_vector_sequence = []

    for l in range(level+1):

        knot_vector_sequence_level = list(data_sequence[l])

        if (l > 0):

            i1 = position_find(knot_vector_sequence[l-1],data_sequence[l][0])

            knot_vector_sequence_level.insert(0,knot_vector_sequence[l-1][i1])
            knot_vector_sequence_level.append(knot_vector_sequence[l-1][i1+1])

        knot_vector_sequence.append(knot_vector_sequence_level)

    return knot_vector_sequence

def knot_vector_sum(knot_vector_block_total,knot_vector_block,den):

    if (len(knot_vector_block_total) == 0):

        knot_vector_block_total = []

        for x in knot_vector_block:

            knot_vector_block_total.append(x/den)

    else:

        for n in range(len(knot_vector_block_total)):

            knot_vector_block_total[n] = knot_vector_block_total[n]+knot_vector_block[n]/den

    return knot_vector_block_total

def knot_vector_repeat(knot_vector,order=3):

    max_knot_vector = max(knot_vector)
    min_knot_vector = min(knot_vector)
    
    knot_vector = list(knot_vector)

    for k in range(order):
        
        knot_vector.insert(0,min_knot_vector)
        knot_vector.append(max_knot_vector)

    return knot_vector

def position_find(knot_vector,vector):

    len_knot = len(knot_vector)

    if (len_knot == 1):

        i = 0

    else:

        for n in range(len_knot-1):

            if ((vector-knot_vector[n]) >= 0 and (vector-knot_vector[n+1]) < 0):

                i = n
                break

            else:

                i = len_knot-1

    return i

def tangent_vector_calculate(data_point,data_point_value):

    len_knot_vector = len(data_point)

    first_tangent_vector = np.zeros(2)
    last_tangent_vector = np.zeros(2)

    first_tangent_vector[0] = data_point[1]-data_point[0]
    last_tangent_vector[0] = data_point[len_knot_vector-1]-data_point[len_knot_vector-2]
    first_tangent_vector[1] = data_point_value[1]-data_point_value[0]
    last_tangent_vector[1] = data_point_value[len_knot_vector-1]-data_point_value[len_knot_vector-2]

    return first_tangent_vector,last_tangent_vector

def vector_position_find(knot_vector,vector):

    knot_vector_level = len(knot_vector)

    min_dif = max(knot_vector[0][0])
    max_dif = -max(knot_vector[0][0])

    for l in range(knot_vector_level):

        len_knot_vector_block = len(knot_vector[l])

        for b in range(len_knot_vector_block):

            len_knot_vector = len(knot_vector[l][b])

            for n in range(len_knot_vector):

                dif = vector-knot_vector[l][b][n]

                if (dif >= 0 and dif <= min_dif):

                    min_dif = dif

                    left_number = n
                    left_block = b
                    left_level = l

                elif (dif < 0 and dif > max_dif):

                    max_dif = dif

                    right_number = n
                    right_block = b
                    right_level = l

    return left_number,left_block,left_level,right_number,right_block,right_level

def vector_sequence_find(knot_vector,vector):

    knot_vector_sequence = []

    vector_position = vector_position_find(knot_vector,vector)

    if (vector_position[1] >= vector_position[3]):

        knot_vector_number = vector_position[0]
        knot_vector_block = vector_position[1]
        knot_vector_level = vector_position[2]
        level = vector_position[2]        

    else:

        knot_vector_number = vector_position[3]
        knot_vector_block = vector_position[4]
        knot_vector_level = vector_position[5]
        level = vector_position[5]        

    for l in range(level):

        knot_vector_number_sequence_level = [knot_vector_number]
        knot_vector_sequence_level = [knot_vector[knot_vector_level][knot_vector_number]]        

        len_knot_vector_level = len(knot_vector[knot_vector_level])

        vector_position = data_point_position_find(knot_vector,knot_vector_number,knot_vector_level)

        left_number = vector_position[0]
        left_block = vector_position[1]
        left_level = vector_position[2]
        right_number = vector_position[3]
        right_block = vector_position[4]
        right_level = vector_position[5]
            
        if (vector_position[2] == knot_vector_level or vector_position[5] == knot_vector_level):

            if (vector_position[2] == knot_vector_level):

                if vector_position[0] not in knot_vector_number_sequence_level:

                    knot_vector_number_sequence_level.insert(0,vector_position[0])
                    knot_vector_sequence_level.insert(0,knot_vector[knot_vector_level][vector_position[0]])

                for n in range(len_knot_vector_level):

                    vector_position = data_point_position_find(knot_vector,vector_position[0],vector_position[1])

                    left_number = vector_position[0]
                    left_level = vector_position[1]

                    if (vector_position[1] == knot_vector_level):

                        if vector_position[0] not in knot_vector_number_sequence_level:

                            knot_vector_number_sequence_level.insert(0,vector_position[0])
                            knot_vector_sequence_level.insert(0,knot_vector[knot_vector_level][vector_position[0]])

                    else:

                        break

            if(vector_position[3] == knot_vector_level):

                if vector_position[2] not in knot_vector_number_sequence_level:

                    knot_vector_number_sequence_level.append(vector_position[2])
                    knot_vector_sequence_level.append(knot_vector[knot_vector_level][vector_position[2]])

                for n in range(len_knot_vector_level):

                    vector_position = data_point_position_find(knot_vector,vector_position[2],vector_position[3])

                    right_number = vector_position[2]
                    right_level = vector_position[3]

                    if (vector_position[3] == knot_vector_level):

                        if vector_position[2] not in knot_vector_number_sequence_level:

                            knot_vector_number_sequence_level.append(vector_position[2])
                            knot_vector_sequence_level.append(knot_vector[knot_vector_level][vector_position[2]])

                    else:

                        break

        if (left_level >= right_level):

            knot_vector_number = left_number
            knot_vector_level = left_level

        elif (left_level < right_level):

            knot_vector_number = right_number
            knot_vector_level = right_level

        knot_vector_sequence.insert(0,knot_vector_sequence_level)

    knot_vector_sequence.insert(0,knot_vector[0])
        
    return knot_vector_sequence