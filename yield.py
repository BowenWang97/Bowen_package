import math
import numpy as np

def yield_calculate_1d(parameter,parameter_delta,parameter_variance,performance,performance_condition):

    len_parameter = len(parameter)

    parameter_start = parameter[0]+2*parameter_variance
    parameter_stop = parameter[len_parameter-1]-2*parameter_variance

    len_yield_parameter = int((parameter_stop-parameter_start)/parameter_delta)+1

    number_dif = int(2*parameter_variance/parameter_delta)

    yield_parameter = np.zeros(len_yield_parameter)
    yield_performance = np.zeros(len_yield_parameter)

    for i in range(len_yield_parameter):

        yield_parameter[i] = parameter[i+number_dif]

        yield_sum = 0
        distribution_sum = 0

        for j in range(2*number_dif+1):

            gaussian = math.exp(-(parameter[i+j]-yield_parameter[i])*(parameter[i+j]-yield_parameter[i])/(2*parameter_variance*parameter_variance))

            distribution_sum = distribution_sum+gaussian

            if (performance[i+j] >= performance_condition):

                yield_sum = yield_sum+gaussian

        yield_performance[i] = yield_sum/distribution_sum

    return(yield_parameter,yield_performance)

def yield_calculate_2d(parameter_1,parameter_1_delta,parameter_1_variance,parameter_2,parameter_2_delta,parameter_2_variance,performance,performance_condition):

    len_parameter_1 = len(parameter_1)
    len_parameter_2 = len(parameter_2)

    parameter_1_start = parameter_1[0]+2*parameter_1_variance
    parameter_1_stop = parameter_1[len_parameter_1-1]-2*parameter_1_variance
    parameter_2_start = parameter_2[0]+2*parameter_2_variance
    parameter_2_stop = parameter_2[len_parameter_2-1]-2*parameter_2_variance

    len_yield_parameter_1 = int((parameter_1_stop-parameter_1_start)/parameter_1_delta)+1
    len_yield_parameter_2 = int((parameter_2_stop-parameter_2_start)/parameter_2_delta)+1

    number_dif_1 = int(2*parameter_1_variance/parameter_1_delta)
    number_dif_2 = int(2*parameter_2_variance/parameter_2_delta)

    yield_parameter_1 = np.zeros(len_yield_parameter_1)
    yield_parameter_2 = np.zeros(len_yield_parameter_2)
    yield_performance = np.zeros(shape = (len_yield_parameter_1,len_yield_parameter_2))

    for i1 in range(len_yield_parameter_1):

        yield_parameter_1[i1] = parameter_1[i1+number_dif_1]

        for i2 in range(len_yield_parameter_2):

            yield_parameter_2[i2] = parameter_2[i2+number_dif_2]

            yield_sum = 0
            distribution_sum = 0

            for j1 in range(2*number_dif_1+1):

                for j2 in range(2*number_dif_2+1):

                    gaussian = math.exp(-(parameter_1[i1+j1]-yield_parameter_1[i1])*(parameter_1[i1+j1]-yield_parameter_1[i1])/(2*parameter_1_variance*parameter_1_variance)-(parameter_2[i2+j2]-yield_parameter_2[i2])*(parameter_2[i2+j2]-yield_parameter_2[i2])/(2*parameter_2_variance*parameter_2_variance))

                    distribution_sum = distribution_sum+gaussian

                    if (performance[i1+j1][i2+j2] >= performance_condition):

                        yield_sum = yield_sum+gaussian
                
            yield_performance[i1][i2] = yield_sum/distribution_sum

    return(yield_parameter_1,yield_parameter_2,yield_performance)

def yield_calculate_3d(parameter_1,parameter_1_delta,parameter_1_variance,parameter_2,parameter_2_delta,parameter_2_variance,parameter_3,parameter_3_delta,parameter_3_variance,performance,performance_condition):

    len_parameter_1 = len(parameter_1)
    len_parameter_2 = len(parameter_2)
    len_parameter_3 = len(parameter_3)

    parameter_1_start = parameter_1[0]+2*parameter_1_variance
    parameter_1_stop = parameter_1[len_parameter_1-1]-2*parameter_1_variance
    parameter_2_start = parameter_2[0]+2*parameter_2_variance
    parameter_2_stop = parameter_2[len_parameter_2-1]-2*parameter_2_variance
    parameter_3_start = parameter_3[0]+2*parameter_3_variance
    parameter_3_stop = parameter_3[len_parameter_3-1]-2*parameter_3_variance

    len_yield_parameter_1 = int((parameter_1_stop-parameter_1_start)/parameter_1_delta)+1
    len_yield_parameter_2 = int((parameter_2_stop-parameter_2_start)/parameter_2_delta)+1
    len_yield_parameter_3 = int((parameter_3_stop-parameter_3_start)/parameter_3_delta)+1

    number_dif_1 = int(2*parameter_1_variance/parameter_1_delta)
    number_dif_2 = int(2*parameter_2_variance/parameter_2_delta)
    number_dif_3 = int(2*parameter_3_variance/parameter_3_delta)

    yield_parameter_1 = np.zeros(len_yield_parameter_1)
    yield_parameter_2 = np.zeros(len_yield_parameter_2)
    yield_parameter_3 = np.zeros(len_yield_parameter_3)
    yield_performance = np.zeros(shape = (len_yield_parameter_1,len_yield_parameter_2,len_yield_parameter_3))

    for i1 in range(len_yield_parameter_1):

        yield_parameter_1[i1] = parameter_1[i1+number_dif_1]

        for i2 in range(len_yield_parameter_2):

            yield_parameter_2[i2] = parameter_2[i2+number_dif_2]

            for i3 in range(len_yield_parameter_3):

                yield_parameter_3[i3] = parameter_3[i3+number_dif_3]

                yield_sum = 0
                distribution_sum = 0

                for j1 in range(2*number_dif_1+1):

                    for j2 in range(2*number_dif_2+1):

                        for j3 in range(2*number_dif_3+1):

                            gaussian = math.exp(-(parameter_1[i1+j1]-yield_parameter_1[i1])*(parameter_1[i1+j1]-yield_parameter_1[i1])/(2*parameter_1_variance*parameter_1_variance)-(parameter_2[i2+j2]-yield_parameter_2[i2])*(parameter_2[i2+j2]-yield_parameter_2[i2])/(2*parameter_2_variance*parameter_2_variance)-(parameter_3[i3+j3]-yield_parameter_3[i3])*(parameter_3[i3+j3]-yield_parameter_3[i3])/(2*parameter_3_variance*parameter_3_variance))

                            distribution_sum = distribution_sum+gaussian

                            if (performance[i1+j1][i2+j2][i3+j3] >= performance_condition):

                                yield_sum = yield_sum+gaussian
                    
                yield_performance[i1][i2][i3] = yield_sum/distribution_sum

    return(yield_parameter_1,yield_parameter_2,yield_parameter_3,yield_performance)

def yield_calculate_4d(parameter_1,parameter_1_delta,parameter_1_variance,parameter_2,parameter_2_delta,parameter_2_variance,parameter_3,parameter_3_delta,parameter_3_variance,parameter_4,parameter_4_delta,parameter_4_variance,performance,performance_condition):

    len_parameter_1 = len(parameter_1)
    len_parameter_2 = len(parameter_2)
    len_parameter_3 = len(parameter_3)
    len_parameter_4 = len(parameter_4)

    parameter_1_start = parameter_1[0]+2*parameter_1_variance
    parameter_1_stop = parameter_1[len_parameter_1-1]-2*parameter_1_variance
    parameter_2_start = parameter_2[0]+2*parameter_2_variance
    parameter_2_stop = parameter_2[len_parameter_2-1]-2*parameter_2_variance
    parameter_3_start = parameter_3[0]+2*parameter_3_variance
    parameter_3_stop = parameter_3[len_parameter_3-1]-2*parameter_3_variance
    parameter_4_start = parameter_4[0]+2*parameter_4_variance
    parameter_4_stop = parameter_4[len_parameter_4-1]-2*parameter_4_variance

    len_yield_parameter_1 = int((parameter_1_stop-parameter_1_start)/parameter_1_delta)+1
    len_yield_parameter_2 = int((parameter_2_stop-parameter_2_start)/parameter_2_delta)+1
    len_yield_parameter_3 = int((parameter_3_stop-parameter_3_start)/parameter_3_delta)+1
    len_yield_parameter_4 = int((parameter_4_stop-parameter_4_start)/parameter_4_delta)+1

    print(len_yield_parameter_1)
    print(len_yield_parameter_2)
    print(len_yield_parameter_3)
    print(len_yield_parameter_4)    

    number_dif_1 = int(2*parameter_1_variance/parameter_1_delta)
    number_dif_2 = int(2*parameter_2_variance/parameter_2_delta)
    number_dif_3 = int(2*parameter_3_variance/parameter_3_delta)
    number_dif_4 = int(2*parameter_4_variance/parameter_4_delta)

    print(2*number_dif_1+1)
    print(2*number_dif_2+1)
    print(2*number_dif_3+1)
    print(2*number_dif_4+1)

    yield_parameter_1 = np.zeros(len_yield_parameter_1)
    yield_parameter_2 = np.zeros(len_yield_parameter_2)
    yield_parameter_3 = np.zeros(len_yield_parameter_3)
    yield_parameter_4 = np.zeros(len_yield_parameter_4)
    yield_performance = np.zeros(shape = (len_yield_parameter_1,len_yield_parameter_2,len_yield_parameter_3,len_yield_parameter_4))

    for i1 in range(len_yield_parameter_1):

        print("i1 =",i1)

        yield_parameter_1[i1] = parameter_1[i1+number_dif_1]

        for i2 in range(len_yield_parameter_2):

            print("i2 =",i2)
            
            yield_parameter_2[i2] = parameter_2[i2+number_dif_2]

            for i3 in range(len_yield_parameter_3):

                yield_parameter_3[i3] = parameter_3[i3+number_dif_3]

                for i4 in range(len_yield_parameter_4):

                    yield_parameter_4[i4] = parameter_4[i4+number_dif_4]

                    yield_sum = 0
                    distribution_sum = 0

                    for j1 in range(2*number_dif_1+1):

                        for j2 in range(2*number_dif_2+1):

                            for j3 in range(2*number_dif_3+1):

                                for j4 in range(2*number_dif_4):

                                    gaussian = math.exp(-(parameter_1[i1+j1]-yield_parameter_1[i1])*(parameter_1[i1+j1]-yield_parameter_1[i1])/(2*parameter_1_variance*parameter_1_variance)-(parameter_2[i2+j2]-yield_parameter_2[i2])*(parameter_2[i2+j2]-yield_parameter_2[i2])/(2*parameter_2_variance*parameter_2_variance)-(parameter_3[i3+j3]-yield_parameter_3[i3])*(parameter_3[i3+j3]-yield_parameter_3[i3])/(2*parameter_3_variance*parameter_3_variance)-(parameter_4[i4+j4]-yield_parameter_4[i4])*(parameter_4[i4+j4]-yield_parameter_4[i4])/(2*parameter_4_variance*parameter_4_variance))

                                    distribution_sum = distribution_sum+gaussian

                                    if (performance[i1+j1][i2+j2][i3+j3][i4+j4] >= performance_condition):

                                        yield_sum = yield_sum+gaussian
                        
                    yield_performance[i1][i2][i3][i4] = yield_sum/distribution_sum

    return(yield_parameter_1,yield_parameter_2,yield_parameter_3,yield_parameter_4,yield_performance)

def yield_calculate_4d_2(parameter_1,parameter_1_delta,parameter_1_variance,parameter_2,parameter_2_delta,parameter_2_variance,parameter_3,parameter_3_delta,parameter_3_variance,parameter_4,parameter_4_delta,parameter_4_variance,performance_1,performance_condition_1,performance_2,performance_condition_2):

    len_parameter_1 = len(parameter_1)
    len_parameter_2 = len(parameter_2)
    len_parameter_3 = len(parameter_3)
    len_parameter_4 = len(parameter_4)

    parameter_1_start = parameter_1[0]+2*parameter_1_variance
    parameter_1_stop = parameter_1[len_parameter_1-1]-2*parameter_1_variance
    parameter_2_start = parameter_2[0]+2*parameter_2_variance
    parameter_2_stop = parameter_2[len_parameter_2-1]-2*parameter_2_variance
    parameter_3_start = parameter_3[0]+2*parameter_3_variance
    parameter_3_stop = parameter_3[len_parameter_3-1]-2*parameter_3_variance
    parameter_4_start = parameter_4[0]+2*parameter_4_variance
    parameter_4_stop = parameter_4[len_parameter_4-1]-2*parameter_4_variance

    len_yield_parameter_1 = int((parameter_1_stop-parameter_1_start)/parameter_1_delta)+1
    len_yield_parameter_2 = int((parameter_2_stop-parameter_2_start)/parameter_2_delta)+1
    len_yield_parameter_3 = int((parameter_3_stop-parameter_3_start)/parameter_3_delta)+1
    len_yield_parameter_4 = int((parameter_4_stop-parameter_4_start)/parameter_4_delta)+1

    print(len_yield_parameter_1)
    print(len_yield_parameter_2)
    print(len_yield_parameter_3)
    print(len_yield_parameter_4)    

    number_dif_1 = int(2*parameter_1_variance/parameter_1_delta)
    number_dif_2 = int(2*parameter_2_variance/parameter_2_delta)
    number_dif_3 = int(2*parameter_3_variance/parameter_3_delta)
    number_dif_4 = int(2*parameter_4_variance/parameter_4_delta)

    print(2*number_dif_1+1)
    print(2*number_dif_2+1)
    print(2*number_dif_3+1)
    print(2*number_dif_4+1)

    yield_parameter_1 = np.zeros(len_yield_parameter_1)
    yield_parameter_2 = np.zeros(len_yield_parameter_2)
    yield_parameter_3 = np.zeros(len_yield_parameter_3)
    yield_parameter_4 = np.zeros(len_yield_parameter_4)
    yield_performance = np.zeros(shape = (len_yield_parameter_1,len_yield_parameter_2,len_yield_parameter_3,len_yield_parameter_4))

    for i1 in range(len_yield_parameter_1):

        print("i1 =",i1)

        yield_parameter_1[i1] = parameter_1[i1+number_dif_1]

        for i2 in range(len_yield_parameter_2):

            print("i2 =",i2)
            
            yield_parameter_2[i2] = parameter_2[i2+number_dif_2]

            for i3 in range(len_yield_parameter_3):

                yield_parameter_3[i3] = parameter_3[i3+number_dif_3]

                for i4 in range(len_yield_parameter_4):

                    yield_parameter_4[i4] = parameter_4[i4+number_dif_4]

                    yield_sum = 0
                    distribution_sum = 0

                    for j1 in range(2*number_dif_1+1):

                        for j2 in range(2*number_dif_2+1):

                            for j3 in range(2*number_dif_3+1):

                                for j4 in range(2*number_dif_4):

                                    gaussian = math.exp(-(parameter_1[i1+j1]-yield_parameter_1[i1])*(parameter_1[i1+j1]-yield_parameter_1[i1])/(2*parameter_1_variance*parameter_1_variance)-(parameter_2[i2+j2]-yield_parameter_2[i2])*(parameter_2[i2+j2]-yield_parameter_2[i2])/(2*parameter_2_variance*parameter_2_variance)-(parameter_3[i3+j3]-yield_parameter_3[i3])*(parameter_3[i3+j3]-yield_parameter_3[i3])/(2*parameter_3_variance*parameter_3_variance)-(parameter_4[i4+j4]-yield_parameter_4[i4])*(parameter_4[i4+j4]-yield_parameter_4[i4])/(2*parameter_4_variance*parameter_4_variance))

                                    distribution_sum = distribution_sum+gaussian

                                    if (performance_1[i1+j1][i2+j2][i3+j3][i4+j4] >= performance_condition_1 and performance_2[i1+j1][i2+j2][i3+j3][i4+j4] >= performance_condition_2):

                                        yield_sum = yield_sum+gaussian
                        
                    yield_performance[i1][i2][i3][i4] = yield_sum/distribution_sum

    return(yield_parameter_1,yield_parameter_2,yield_parameter_3,yield_parameter_4,yield_performance)