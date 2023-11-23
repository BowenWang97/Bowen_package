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

                B_spline[k] = (knot_vector[n+k+1]-vector)/(knot_vector[n+k+1]-knot_vector[n+1])*B_spline[n+1][k-1]

            elif (knot_vector[n+k]-knot_vector[n] != 0 and (knot_vector[n+k+1]-knot_vector[n+1]) == 0):

                B_spline[k] = (vector-knot_vector[n])/(knot_vector[n+k]-knot_vector[n])*B_spline[k-1]

            elif (knot_vector[n+k]-knot_vector[n] == 0 and (knot_vector[n+k+1]-knot_vector[n+1]) == 0):

                B_spline[n][k] = 0

            else:

                B_spline[n][k] = (vector-knot_vector[n])/(knot_vector[n+k]-knot_vector[n])*B_spline[n][k-1]+(knot_vector[n+k+1]-vector)/(knot_vector[n+k+1]-knot_vector[n+1])*B_spline[n+1][k-1]

    return B_spline

def B_spline_split(B_spline_1,B_spline_2,knot_vector):

    len_knot_vector = len(knot_vector)

    BS_split = np.zeros(shape = (len_knot_vector,order+1))

    BS_split[0][0] = B_spline_1

def control_point_calculate_2d(data_point,data_point_value,first_tangent_vector,last_tangent_vector,knot_vector,order=3):

    level_data_point = len(data_point)

    for l in range(level_data_point):

        if (l == 0):

            len_data_point = len(data_point[0])

            control_point = np.zeros(len_data_point+2)
            control_point_value = np.zeros(len_data_point+2)

            max_knot_vector = max(knot_vector[0])

            knot_vector_level = knot_vector_repeat(knot_vector[0],order)

            control_point[0] = data_point[0][0]
            control_point_value[0] = data_point_value[0][0]
            control_point[1] = control_point[0]+knot_vector_level[1+order]*first_tangent_vector[0]/order
            control_point_value[1] = control_point_value[0]+knot_vector_level[1+order]*first_tangent_vector[1]/order
            control_point[len_data_point+1] = data_point[0][len_data_point-1]
            control_point_value[len_data_point+1] = data_point_value[0][len_data_point-1]
            control_point[len_data_point] = control_point[len_data_point+1]-(max_knot_vector-knot_vector_level[len_data_point+order-2])*last_tangent_vector[0]/order
            control_point_value[len_data_point] = control_point_value[len_data_point+1]-(max_knot_vector-knot_vector_level[len_data_point+order-2])*last_tangent_vector[1]/order

            data_point_matrix = np.delete(data_point[0],len_data_point-1,0)
            data_point_value_matrix = np.delete(data_point_value[0],len_data_point-1,0)
            data_point_matrix = np.delete(data_point_matrix,0,0)
            data_point_value_matrix = np.delete(data_point_value_matrix,0,0)

            len_data_point_matrix = len(data_point_matrix)

            control_point_matrix = np.zeros(len_data_point_matrix)
            control_point_value_matrix = np.zeros(len_data_point_matrix)
            coefficiency_matrix = np.zeros(shape = (len_data_point_matrix,len_data_point_matrix))

            for m in range(len_data_point_matrix):

                B_spline = B_spline_calculate(knot_vector_level,knot_vector_level[m+order+1],order)

                if (m == 0):

                    data_point_matrix[m] = data_point_matrix[m]-B_spline[m+1][order]*control_point[1]
                    data_point_value_matrix[m] = data_point_value_matrix[m]-B_spline[m+1][order]*control_point_value[1]

                    for i in range(order-1):

                        coefficiency_matrix[m][m+i] = B_spline[m+i+2][order]

                elif (m == len_data_point_matrix-1):

                    data_point_matrix[m] = data_point_matrix[m]-B_spline[m+3][order]*control_point[len_data_point]
                    data_point_value_matrix[m] = data_point_value_matrix[m]-B_spline[m+3][order]*control_point_value[len_data_point]

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

            control_point_matrix = np.dot(coefficiency_matrix_inv,data_point_matrix)
            control_point_value_matrix = np.dot(coefficiency_matrix_inv,data_point_value_matrix)

            for m in range(2,len_data_point):

                control_point[m] = control_point_matrix[m-2]
                control_point_value[m] = control_point_value_matrix[m-2]

            control_point = [control_point]
            control_point_value = [control_point_value]

        else:

            len_data_point = len(data_point[l])

            control_point_level = np.zeros(len_data_point)
            control_point_value_level = np.zeros(len_data_point)

            control_point.append(control_point_level)
            control_point_value.append(control_point_value_level)

            for n in range(len_data_point):

                THB_spline = LRB_spline_calculate(knot_vector,knot_vector[l][n],order)

                knot_vector_level = knot_vector_summary(knot_vector,n,l)
                
                control_point_summary = control_point_summary_2d(control_point,control_point_value,knot_vector,n,l)

                control_point_value_summary = control_point_summary[1]
                control_point_summary = control_point_summary[0]

                i = position_find(knot_vector_level,knot_vector[l][n])

                control_point[l][n] = (data_point[l][n]-control_point_summary[i+1]*THB_spline[i+1]-control_point_summary[i+2]*THB_spline[i+2])/(THB_spline[i])
                control_point_value[l][n] = (data_point_value[l][n]-control_point_value_summary[i+1]*THB_spline[i+1]-control_point_value_summary[i+2]*THB_spline[i+2])/(THB_spline[i])

    return control_point,control_point_value

def control_point_summary_2d(control_point,control_point_value,knot_vector,knot_vector_number,level):

    knot_vector_sequence = data_sequence_find(knot_vector,knot_vector_number,level)

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

def data_point_derivative_calculate(all_data_point,all_data_point_value,all_knot_vector,derivative_order=1):

    len_all_data_point = len(all_data_point)

    data_point_derivative = np.zeros(shape = (len_all_data_point,derivative_order+1))
    data_point_value_derivative = np.zeros(shape = (len_all_data_point,derivative_order+1))

    knot_vector_derovative = knot_vector_derivative_calculate(all_knot_vector,derivative_order-1)

    for k in range(derivative_order+1):

        if (k == 0):

            for m in range(len_all_data_point-k):

                data_point_derivative[m][k] = all_data_point[m]

        else:

            for m in range(len_all_data_point-k):

                data_point_derivative[m][k] = (data_point_derivative[m+1][k-1]-data_point_derivative[m][k-1])/(knot_vector_derovative[m+1][k-1]-knot_vector_derovative[m][k-1])

    for k in range(derivative_order+1):

        if (k == 0):

            for m in range(len_all_data_point-k):

                data_point_value_derivative[m][k] = all_data_point_value[m]

        else:

            for m in range(len_all_data_point-k):

                data_point_value_derivative[m][k] = (data_point_value_derivative[m+1][k-1]-data_point_value_derivative[m][k-1])/(knot_vector_derovative[m+1][k-1]-knot_vector_derovative[m][k-1])
    
    return data_point_derivative,data_point_value_derivative

def data_point_distance_calculate(data_point,data_point_value):

    data_point_distance = math.sqrt((data_point[0]-data_point[1])*(data_point[0]-data_point[1])+(data_point_value[0]-data_point_value[1])*(data_point_value[0]-data_point_value[1]))

    return data_point_distance

def data_point_position_find(data_point,data_point_number,data_point_level):

    min_dif = max(data_point[0])-min(data_point[0])
    max_dif = min(data_point[0])-max(data_point[0])

    for l in range(data_point_level+1):

        len_data_point = len(data_point[l])

        for n in range(len_data_point):

            dif = data_point[data_point_level][data_point_number]-data_point[l][n]

            if (dif > 0 and dif < min_dif):

                min_dif = dif

                left_number = n
                left_level = l

            elif (dif < 0 and dif > max_dif):

                max_dif = dif

                right_number = n
                right_level = l

    return left_number,left_level,right_number,right_level

def data_sequence_find(data_point,number,level):

    data_point_sequence = []
    data_point_number_sequence = []

    data_point_number = number
    data_point_level = level

    for l in range(level):

        data_point_number_sequence_level = [data_point_number]
        data_point_sequence_level = [data_point[data_point_level][data_point_number]]        

        len_data_point_level = len(data_point[data_point_level])

        data_point_position = data_point_position_find(data_point,data_point_number,data_point_level)

        left_number = data_point_position[0]
        left_level = data_point_position[1]
        right_number = data_point_position[2]
        right_level = data_point_position[3]
            
        if (data_point_position[1] == data_point_level or data_point_position[3] == data_point_level):

            if (data_point_position[1] == data_point_level):

                if data_point_position[0] not in data_point_number_sequence_level:

                    data_point_number_sequence_level.insert(0,data_point_position[0])
                    data_point_sequence_level.insert(0,data_point[data_point_level][data_point_position[0]])

                for n in range(len_data_point_level):

                    data_point_position = data_point_position_find(data_point,data_point_position[0],data_point_position[1])

                    left_number = data_point_position[0]
                    left_level = data_point_position[1]

                    if (data_point_position[1] == data_point_level):

                        if data_point_position[0] not in data_point_number_sequence_level:

                            data_point_number_sequence_level.insert(0,data_point_position[0])
                            data_point_sequence_level.insert(0,data_point[data_point_level][data_point_position[0]])

                    else:

                        break

            if(data_point_position[3] == data_point_level):

                if data_point_position[2] not in data_point_number_sequence_level:

                    data_point_number_sequence_level.append(data_point_position[2])
                    data_point_sequence_level.append(data_point[data_point_level][data_point_position[2]])

                for n in range(len_data_point_level):

                    data_point_position = data_point_position_find(data_point,data_point_position[2],data_point_position[3])

                    right_number = data_point_position[2]
                    right_level = data_point_position[3]

                    if (data_point_position[3] == data_point_level):

                        if data_point_position[2] not in data_point_number_sequence_level:

                            data_point_number_sequence_level.append(data_point_position[2])
                            data_point_sequence_level.append(data_point[data_point_level][data_point_position[2]])

                    else:

                        break

        if (left_level >= right_level):

            data_point_number = left_number
            data_point_level = left_level

        elif (left_level < right_level):

            data_point_number = right_number
            data_point_level = right_level

        data_point_sequence.insert(0,data_point_sequence_level)
        data_point_number_sequence.insert(0,data_point_number_sequence_level)

    data_point_number_sequence_level = [left_number,right_number]
    data_point_sequence_level = [data_point[0][left_number],data_point[0][right_number]]

    data_point_sequence.insert(0,data_point_sequence_level)
    data_point_number_sequence.insert(0,data_point_number_sequence_level)
        
    return data_point_sequence,data_point_number_sequence

def data_summary_2d(data,data_point,level):

    len_data = len(data)

    data_summary = []

    for m in range(len_data):

        all_data_point = list(data_point[0])
        data_level = list(data[m][0])

        for l in range(1,level):

            len_data_point = len(data_point[l])

            for n in range(len_data_point):

                i = position_find(all_data_point,data_point[l][n])

                all_data_point.insert(i+1,data_point[l][n])
                data_level.insert(i+1,data[m][l][n])

        data_summary.append(data_level)

    return data_summary

def fit_2d(data_point,data_point_value,point_delta,order=3):

    vector_delta = point_delta/50

    knot_vector = knot_vector_calculate_2d(data_point,data_point_value)

    level_data_point = len(data_point)

    data_summary = data_summary_2d([data_point,data_point_value,knot_vector],data_point,level_data_point)

    all_data_point = data_summary[0]
    all_data_point_value = data_summary[1]
    all_knot_vector = data_summary[2]

    len_all_data_point = len(all_data_point)

    data_point_derivative = data_point_derivative_calculate(all_data_point,all_data_point_value,all_knot_vector)

    first_tangent_vector = [data_point_derivative[0][0][1],data_point_derivative[1][0][1]]
    last_tangent_vector = [data_point_derivative[0][len_all_data_point-order+1][1],data_point_derivative[1][len_all_data_point-order+1][1]]

    control_point = control_point_calculate_2d(data_point,data_point_value,first_tangent_vector,last_tangent_vector,knot_vector,order)

    point_number = int((max(data_point[0])-min(data_point[0]))/point_delta)
    fit_point = np.zeros(shape = (2,point_number+1))

    for n in range(point_number+1):

        NUTHB_spline_point = LRB_spline_point_calculate_2d(control_point,knot_vector,point_vector[n],order)

        if (n == point_number):

            for d in range(2):

                fit_point[d][n] = data_point[d][len_data_point-1]

        else:

            for d in range(2):

                fit_point[d][n] = NUTHB_spline_point[d]

    return fit_point

def LRHB_spline_calculate(knot_vector,vector,order=3):

    vector_position = vector_position_find(knot_vector,vector)

    if (vector_position[1] >= vector_position[3]):

        knot_vector_number = vector_position[0]
        level = vector_position[1]        

    else:

        knot_vector_number = vector_position[2]
        level = vector_position[3]   

    knot_vector_sequence = data_sequence_find(knot_vector,knot_vector_number,level)

    knot_vector_sequence = knot_vector_sequence[0]

    knot_vector_0 = knot_vector_repeat(knot_vector[0],order)

    THB_spline = [B_spline_calculate(knot_vector_0,vector,order)[:,order]]

    for l in range(1,level+1):

        i1 = position_find(knot_vector_sequence[l-1],knot_vector_sequence[l][0])

        knot_vector_l = knot_vector_sequence[l]

        knot_vector_l.insert(0,knot_vector_sequence[l-1][i1])
        knot_vector_l.append(knot_vector_sequence[l-1][i1])

        B_spline = B_spline_calculate(knot_vector_summary,vector,order)

        THB_spline_level = THB_spline[i]*B_spline[i+1][order]

        THB_spline.insert(i+1,THB_spline_level)

        THB_spline[i] = THB_spline[i]*B_spline[i][order]

    return THB_spline

def LRHB_spline_point_calculate_2d(control_point,control_point_value,knot_vector,vector,order=3):

    level_knot_vector = len(knot_vector)

    vector_position = vector_position_find(knot_vector,vector)

    control_summary = control_point_summary_2d([control_point,control_point_value,knot_vector],knot_vector,max(vector_position[1],vector_position[3]))

    control_point_level = control_summary[0][0]
    control_point_value_level = control_summary[0][1]
    knot_vector_level = control_summary[1]

    len_control_point_level = len(control_point_level)

    knot_vector_level = knot_vector_repeat(knot_vector_level,order)

    point = 0
    point_value = 0

    i = position_find(knot_vector_level,vector)

    B_spline = B_spline_calculate(knot_vector_level,vector,order)

    for m in range(order+1):

        if (i+m >= len_control_point_level):

            continue

        else:

            point = point+control_point_level[i+m]*B_spline[i+m][order]
            point_value = point_value+control_point_value_level[i+m]*B_spline[i+m][order]

    return point,point_value

def knot_vector_calculate_2d(data_point,data_point_value):

    level_data_point = len(data_point)

    for l in range(level_data_point):

        if (l == 0):

            len_data_point = len(data_point[0])

            knot_vector = np.zeros(len_data_point)

            for n in range(len_data_point-1):

                knot_vector[n+1] = knot_vector[n]+data_point_distance_calculate([data_point[0][n+1],data_point[0][n]],[data_point_value[0][n+1],data_point_value[0][n]])
            
            knot_vector = [knot_vector]

        else:

            len_data_point = len(data_point[l])

            knot_vector_level = np.zeros(len_data_point)

            for n in range(len_data_point):

                data_sequence = data_sequence_find(data_point,n,l)

                data_point_number_sequence = data_sequence[1]
                data_point_sequence = data_sequence[0]

                i1 = position_find(data_point[l-1],data_point[l][n])
                i2 = position_find(data_point[l],data_point[l][n])

                distance_sum = 0

                for nd in range(len(data_point_sequence[l])+1):

                    if (nd == 0):

                        dn = data_point_number_sequence[l][nd]

                        distance_sum = distance_sum+data_point_distance_calculate([data_point[l-1][i1],data_point[l][dn]],[data_point_value[l-1][i1],data_point_value[l][dn]])

                        if (dn == i2):

                            knot_distance = distance_sum
                        
                    elif (nd == len(data_point_sequence[l])):

                        dn = data_point_number_sequence[l][nd-1]

                        distance_sum = distance_sum+data_point_distance_calculate([data_point[l][dn],data_point[l-1][i1+1]],[data_point_value[l][dn],data_point_value[l-1][i1+1]])

                    else:

                        dn = data_point_number_sequence[l][nd]

                        distance_sum = distance_sum+data_point_distance_calculate([data_point[l][dn-1],data_point[l][dn]],[data_point_value[l][dn-1],data_point_value[l][dn]])

                        if (dn == i2):

                            knot_distance = distance_sum

                knot_vector_level[n] = knot_vector[l-1][i1]+(knot_vector[l-1][i1+1]-knot_vector[l-1][i1])*knot_distance/distance_sum

        # elif (l == 1):

        #     len_data_point = len(data_point[1])

        #     knot_vector_level = np.zeros(len_data_point)

        #     for n in range(len_data_point):

        #         i = position_find(data_point[0],data_point[1][n])

        #         knot_vector_level[n] = knot_vector[0][i]+(knot_vector[0][i+1]-knot_vector[0][i])*math.sqrt((data_point[1][n]-data_point[0][i])*(data_point[1][n]-data_point[0][i])+(data_point_value[1][n]-data_point_value[0][i])*(data_point_value[1][n]-data_point_value[0][i]))/(math.sqrt((data_point[1][n]-data_point[0][i])*(data_point[1][n]-data_point[0][i])+(data_point_value[1][n]-data_point_value[0][i])*(data_point_value[1][n]-data_point_value[0][i]))+math.sqrt((data_point[1][n]-data_point[0][i+1])*(data_point[1][n]-data_point[0][i+1])+(data_point_value[1][n]-data_point_value[0][i+1])*(data_point_value[1][n]-data_point_value[0][i+1])))

        #     knot_vector.append(knot_vector_level)

        # else:

        #     len_data_point = len(data_point[l])

        #     knot_vector_level = np.zeros(len_data_point)

        #     for n in range(len_data_point):

        #         data_point_position = data_point_position_find(data_point,n,l)

        #         ln = data_point_position[0]
        #         ll = data_point_position[1]
        #         rn = data_point_position[2]
        #         rl = data_point_position[3]

        #         knot_vector_level[n] = knot_vector[ll][ln]+(knot_vector[rl][rn]-knot_vector[ll][ln])*math.sqrt((data_point[l][n]-data_point[ll][ln])*(data_point[l][n]-data_point[ll][ln])+(data_point_value[l][n]-data_point_value[ll][ln])*(data_point_value[l][n]-data_point_value[ll][ln]))/(math.sqrt((data_point[l][n]-data_point[ll][ln])*(data_point[l][n]-data_point[ll][ln])+(data_point_value[l][n]-data_point_value[ll][ln])*(data_point_value[l][n]-data_point_value[ll][ln]))+math.sqrt((data_point[l][n]-data_point[rl][rn])*(data_point[l][n]-data_point[rl][rn])+(data_point_value[l][n]-data_point_value[rl][rn])*(data_point_value[l][n]-data_point_value[rl][rn])))

            knot_vector.append(knot_vector_level)

    return knot_vector

def knot_vector_derivative_calculate(all_knot_vector,derivative_order):

    len_all_knot_vector = len(all_knot_vector)

    knot_vector_derivative = np.zeros(shape = (len_all_knot_vector,derivative_order+1))

    for k in range(derivative_order+1):

        if (k == 0):
            
            for m in range(len_all_knot_vector-k):

                knot_vector_derivative[m][k] = all_knot_vector[m]

        else:

            for m in range(len_all_knot_vector-k):

                knot_vector_derivative[m][k] = (knot_vector_derivative[m][k-1]+knot_vector_derivative[m+1][k-1])/2

    return knot_vector_derivative

def knot_vector_summary(knot_vector,knot_vector_number,level):

    knot_vector_sequence = data_sequence_find(knot_vector,knot_vector_number,level)

    knot_vector_summary = list(knot_vector[0])

    for l in range(0,level):

        if (l == 0):

            i = position_find(knot_vector_summary,knot_vector[l+1][knot_vector_number])

            knot_vector_summary.insert(i+1,knot_vector[l+1][knot_vector_number])

        else:

            i = position_find(knot_vector_summary,knot_vector[l+1][knot_vector_sequence[l]])

            knot_vector_summary.insert(i+1,knot_vector[l+1][knot_vector_sequence[l]])

    return knot_vector_summary

def knot_vector_repeat(knot_vector,order=3):

    max_knot_vector = max(knot_vector)
    knot_vector = list(knot_vector)

    for k in range(order):
        
        knot_vector.insert(0,0)
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

def vector_position_find(knot_vector,vector):

    level_knot_vector = len(knot_vector)

    min_dif = max(knot_vector[0])
    max_dif = -max(knot_vector[0])

    for l in range(level_knot_vector):

        len_knot_vector = len(knot_vector[l])

        for n in range(len_knot_vector):

            dif = vector-knot_vector[l][n]

            if (dif >= 0 and dif <= min_dif):

                min_dif = dif

                left_number = n
                left_level = l

            elif (dif < 0 and dif > max_dif):

                max_dif = dif

                right_number = n
                right_level = l

    return left_number,left_level,right_number,right_level

def vector_sequence_find(knot_vector,vector):

    knot_vector_sequence = []
    knot_vector_number_sequence = []

    vector_position = vector_position_find(knot_vector,vector)

    if (vector_position[1] >= vector_position[3]):

        knot_vector_number = vector_position[0]
        knot_vector_level = vector_position[1]
        level = vector_position[1]        

    else:

        knot_vector_number = vector_position[2]
        knot_vector_level = vector_position[3]
        level = vector_position[3]        

    for l in range(level):

        knot_vector_number_sequence_level = [knot_vector_number]
        knot_vector_sequence_level = [knot_vector[knot_vector_level][knot_vector_number]]        

        len_knot_vector_level = len(knot_vector[knot_vector_level])

        vector_position = data_point_position_find(knot_vector,knot_vector_number,knot_vector_level)

        left_number = vector_position[0]
        left_level = vector_position[1]
        right_number = vector_position[2]
        right_level = vector_position[3]
            
        if (vector_position[1] == knot_vector_level or vector_position[3] == knot_vector_level):

            if (vector_position[1] == knot_vector_level):

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
                            knot_vector_sequence_level.insert(0,data_point[knot_vector_level][vector_position[0]])

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
        knot_vector_number_sequence.insert(0,knot_vector_number_sequence_level)

    knot_vector_number_sequence_level = [left_number,right_number]
    knot_vector_sequence_level = [knot_vector[0][left_number],knot_vector[0][right_number]]

    knot_vector_sequence.insert(0,knot_vector_sequence_level)
    knot_vector_number_sequence.insert(0,knot_vector_number_sequence_level)
        
    return knot_vector_sequence,knot_vector_number_sequence