import math
import numpy as np
import scipy as sci

def simple_alpha(data):

    alpha = np.std(data)

    if (alpha == 0):

        alpha = 1

    return alpha

def linear(x1,x2,alpha = 1,lamda = 1):

    k = x1*x2

    return k

def matern_nu_1(x1,x2,alpha = 1,lamda = 1):

    k = alpha*alpha*math.exp(-abs(x1-x2)/lamda)

    return k

def matern_nu_3(x1,x2,alpha = 1,lamda = 1):

    k = alpha*alpha*(1+math.sqrt(3)*abs(x1-x2)/lamda)*math.exp(-math.sqrt(3)*abs(x1-x2)/lamda)

    return k

def matern_nu_5(x1,x2,alpha = 1,lamda = 1):

    k = alpha*alpha*(1+math.sqrt(5)*abs(x1-x2)/lamda+5*(x1-x2)*(x1-x2)/3/lamda/lamda)*math.exp(-math.sqrt(5)*abs(x1-x2)/lamda)

    return k

def matern_nu_n(x1,x2,nu,alpha = 1,lamda = 1):

    k = alpha*alpha*pow(2,1-nu)/sci.special.gamma(nu)*pow(math.sqrt(2*nu)*abs(x1-x2)/lamda,nu)*sci.special.jv(nu,math.sqrt(2*nu)*abs(x1-x2)/lamda)

    return k

def squared_exponential(x1,x2,alpha = 1,lamda = 1):

    k = alpha*math.exp(-(x1-x2)*(x1-x2)/2/lamda/lamda)

    return k

all_kernel_function = {
    "linear" : linear,
    "matern_nu_1" : matern_nu_1,
    "matern_nu_3" : matern_nu_3,
    "matern_nu_5" : matern_nu_5,
    "matern_nu_n" : matern_nu_n,
    "squared_exponential" : squared_exponential
}

def gp_mean_function(x,data,kernel_function_name,alpha = 1,sigema_n = 0,lamda = 1):

    len_data = len(data[0])

    vector_1 = np.zeros(len_data)

    for n1 in range(len_data):

        vector_1[n1] = all_kernel_function[kernel_function_name](x,data[0][n1],alpha,lamda)

    matrix_1 = np.zeros(shape = (len_data,len_data))

    for n1 in range(len_data):

        for n2 in range(len_data):

            matrix_1[n1][n2] = all_kernel_function[kernel_function_name](data[0][n1],data[0][n2],alpha,lamda)

    if (sigema_n == 0):

        matrix_2 = np.zeros(shape = (len_data,len_data))

    else:

        matrix_2 = np.diag(sigema_n)

    matrix = matrix_1+matrix_2

    inv_matrix = np.linalg.inv(matrix)

    vector_2 = np.zeros(len_data)

    for n2 in range(len_data):

        vector_2[n2] = data[1][n2]

    row_vector_2 = vector_2.reshape((len_data,1))

    mu = np.dot(np.dot(vector_1,inv_matrix),row_vector_2)

    return mu

def gp_standard_deviation_function(x,data,kernel_function_name,alpha = 1,sigema_n = 0,lamda = 1):

    len_data = len(data[0])

    scalar_1 = all_kernel_function[kernel_function_name](x,x,alpha,lamda)

    vector_1 = np.zeros(len_data)

    for n1 in range(len_data):

        vector_1[n1] = all_kernel_function[kernel_function_name](x,data[0][n1],alpha,lamda)

    matrix_1 = np.zeros(shape = (len_data,len_data))

    for n1 in range(len_data):

        for n2 in range(len_data):

            matrix_1[n1][n2] = all_kernel_function[kernel_function_name](data[0][n1],data[0][n2],alpha,lamda)

    if (sigema_n == 0):

        matrix_2 = np.zeros(shape = (len_data,len_data))

    else:

        matrix_2 = np.diag(sigema_n)

    matrix = matrix_1+matrix_2

    inv_matrix = np.linalg.inv(matrix)

    vector_2 = np.zeros(len_data)

    for n2 in range(len_data):

        vector_2[n2] = all_kernel_function[kernel_function_name](data[0][n2],x,alpha,lamda)

    row_vector_2 = vector_2.reshape((len_data,1))

    sigma = scalar_1-np.dot(np.dot(vector_1,inv_matrix),row_vector_2)

    if (sigma > 0):

        sigma = math.sqrt(sigma)

    else:

        sigma = 0

    return sigma

def gp_mean_point(x,data,kernel_function_name,alpha = 1,sigema_n = 0,lamda = 1):

    len_x = len(x)

    mu_f = np.zeros(len_x)

    for n in range(len_x):

        mu_f[n] = gp_mean_function(x[n],data,kernel_function_name,alpha,sigema_n,lamda)

    return mu_f

def gp_standard_deviation_point(x,data,kernel_function_name,alpha = 1,sigema_n = 0,lamda = 1):

    len_x = len(x)

    sigma_f = np.zeros(len_x)

    for n in range(len_x):

        sigma_f[n] = gp_standard_deviation_function(x[n],data,kernel_function_name,alpha,sigema_n,lamda)

    return sigma_f

def gp_sample_f(mean_point,standard_deviation_point):

    f = np.random.multivariate_normal(mean_point,np.diag(standard_deviation_point))

    return f

def simple_xi(data,iteration_time):

    xi = (max(data)-min(data))/iteration_time
    
    if (xi == 0):

        xi = 1

    return xi

def standardized_improvement(x,data,kernel_function_name,alpha = 1,xi = 0.1,sigema_n = 0,lamda = 1):

    z = (gp_mean_function(x,data,kernel_function_name,alpha,sigema_n,lamda)-max(data[1])-xi)/gp_standard_deviation_function(x,data,kernel_function_name,alpha,sigema_n,lamda)

    return z

def expected_improvement(x,data,kernel_function_name,alpha = 1,xi = 0.1,sigema_n = 0,lamda = 1):

    ei = (gp_mean_function(x,data,kernel_function_name,alpha,sigema_n,lamda)-max(data[1])-xi)*sci.stats.norm.cdf(standardized_improvement(x,data,kernel_function_name,alpha,xi,sigema_n))+gp_standard_deviation_function(x,data,kernel_function_name,alpha,sigema_n,lamda)*sci.stats.norm.pdf(standardized_improvement(x,data,kernel_function_name,alpha,xi,sigema_n,lamda))

    return ei

def probability_of_improvement(x,data,kernel_function_name,alpha = 1,xi = 0.1,sigema_n = 0,lamda = 1):

    pi = sci.stats.norm.cdf(standardized_improvement(x,data,kernel_function_name,alpha,xi,sigema_n,lamda))

    return pi

def upper_confidence_bound(x,data,kernel_function_name,alpha = 1,kappa = 1,sigema_n = 0,lamda = 1):

    ucb = gp_mean_function(x,data,kernel_function_name,alpha,sigema_n,lamda)+kappa*gp_standard_deviation_function(x,data,kernel_function_name,alpha,sigema_n,lamda)

    return ucb

all_acquisition_function = {
    "expected_improvement":expected_improvement,
    "probability_of_improvement":probability_of_improvement,
    "upper_confidence_bound":upper_confidence_bound
}

def acquisition_point(x,data,kernel_function_name,acquisition_function_name,alpha = 1,xi = 0.1,sigema_n = 0,lamda = 1):

    len_x = len(x)

    weight = np.zeros(len_x)

    for n in range(len_x):

        weight[n] = all_acquisition_function[acquisition_function_name](x[n],data,kernel_function_name,alpha,xi,sigema_n,lamda)

    return weight

def next_point(x,data,kernel_function_name,acquisition_function_name,alpha = 1,xi = 0.1,sigema_n = 0,lamda = 1):

    weight = acquisition_point(x,data,kernel_function_name,acquisition_function_name,alpha,xi,sigema_n,lamda)

    weight_0 = max(weight)

    len_x = len(x)

    for n in range(len_x):

        weight[n] = weight[n]/weight_0

    next_x = None
    max_weight = 0    

    for n in range(len_x):

        if (weight[n] > max_weight):

            max_weight = weight[n]

            next_x = x[n]

    return next_x,max_weight

def next_point_add(x,next_x):

    for n in range(len(x)):

        if (x[n] > next_x):

            x.insert(n,next_x)

            break

        elif(n == len(x)-1):

            x.append(next_x)

    return x

def second_max(data):

    data_sort = sorted(data)

    if (len(data) == 2):

        return data_sort[0]
    
    else:

        return data_sort[-2]