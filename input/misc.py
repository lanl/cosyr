# -----------------------------------------
# -  misc routines for CoSyR   -
# -----------------------------------------

import numpy as np

## generate test wavelets using 1D ultra relativistic approx. for retarded angle and different methods
def gen_test_wavelets(scaled_alpha, scaled_chi, n_alpha, n_chi, _gamma=10, unscale_coord=True, flatten = True, num_fields=1, distribution="uniform", retard_angle_approx="none", method="potential_far_field", filter=False, filter_exp="np.abs(wx) > 300.0") :
    #global mpi_rank

    #if (mpi_rank == 0) : print("wavelet parameters:", scaled_alpha, scaled_chi, n_alpha, n_chi, _gamma)

    beta = np.sqrt(1.0-_gamma**(-2.0))

    # generate alpha_axis (scaled alpha for wavelets)    
    if (distribution=="gaussian") :
        # -- start modif for GAUSSIAN distrib --
        alpha_axis = np.linspace(-scaled_alpha/2.0, scaled_alpha/2.0, n_alpha)
        mean = scaled_alpha * 0.5
        # print("mean={}".format(mean))
        dev = 190 # 190
        dist = stats.norm(loc=mean, scale=dev)
        bounds = dist.cdf([0, scaled_alpha])
        pp = np.linspace(*bounds, n_alpha)
        vals = dist.ppf(pp)
        alpha_axis = (vals - mean)
        # print(alpha_axis)
        # # -- end modif --

    if (distribution=="uniform") :
        # -- start modif for UNIFORM distrib --
        peak_factor =  1.0 
        peak_shift = 0.0
        #print("peak_factor=", peak_factor, "peak_shift=", peak_shift)
        #if (np.mod(n_alpha,2) ==1) : peak_shift =  0.5*scaled_alpha/n_alpha
        alpha_axis = np.linspace(-scaled_alpha/2.0, scaled_alpha/2.0, n_alpha)
        shifted_alpha_axis = peak_factor*(alpha_axis + peak_shift)
        alpha_axis = shifted_alpha_axis
        # -- end modif

    if (distribution=="adaptive_1d") :
        # -- start modif for ADAPT distrib --
        alpha_max = (scaled_alpha*0.5)/_gamma**3.0
        number_of_wavelets_pos = int(n_alpha/2.0)  # for positive alpha axis
        number_of_wavelets_neg = n_alpha - number_of_wavelets_pos  # for negative alpha axis

        psi_max=(24.0*alpha_max)**(1.0/3.0)
        psi_max_negative_axis = alpha_max/2.0

        #print(alpha_max, psi_max, psi_max_negative_axis)

        psi_for_pos_axis = np.linspace(0.0, psi_max, number_of_wavelets_pos)
        psi_for_neg_axis = np.linspace(psi_max_negative_axis, 0.0, number_of_wavelets_neg, endpoint=False)

        alpha_axis = np.zeros(n_alpha)
        alpha_axis[:number_of_wavelets_neg] = -psi_for_neg_axis*2.0
        alpha_axis[number_of_wavelets_neg:] = psi_for_pos_axis**3.0/24.0

        alpha_axis *= _gamma**3.0

        #np.set_printoptions(suppress=True)
        #print(shifted_alpha_axis)
        # -- end modif

    chi_axis = np.linspace(-scaled_chi/2.0, scaled_chi/2.0, n_chi)
    alpha = alpha_axis/_gamma**3.0
    if (retard_angle_approx == "ultra_relativistic") :
        psi, eta = retard_angle_1D_ultra_relativistic(alpha)
    else : # etard_angle_approx="none"     
        psi, eta = retarded_angle(alpha, beta, 0.0, 0.0)

    wy, wx = np.meshgrid(chi_axis, alpha_axis)
    field = np.ones_like(wx) 

    # calculate field or potential on the wavelets.
    kernel = np.zeros_like(alpha)
    positive_range = alpha > 0
    non_origin = (np.abs(alpha) > 0.0)
    origin = (alpha == 0.0)
    negative_range = alpha < 0
    
    # Method reference: Huang PRAB 2013 
    if (method == "potential_far_field") :
       # for approximated potential, normalized to e/R, Eq. (20) 
       kernel[positive_range] = 2.0*(3.0*alpha[positive_range])**(-1.0/3.0) 
       kernel[negative_range] = - alpha[negative_range]/8.0     
       field = np.transpose(field.T * kernel)

# Note that for 1D field kernel, the result is not expected to agree with the analytic one         
#     if (method == "far_field") :
#        # for approximated field, normalized to e/R^2, derivative of Eq. (20)
#        kernel[positive_range] = 6.0*(3.0*alpha[positive_range])**(-4.0/3.0)
#        kernel[negative_range] = 1.0/8.0        
#        field = np.transpose(field.T * kernel)

#     if (method == "field") :
#        # true radiation field, normalized to e/R^2, Eq. (19), this is non-singular 
#        field = np.transpose(field.T*(0.5*(beta-np.sin(eta))/(1.0-beta*np.sin(eta))**3.0))

    if (method == "potential") :
       # full lw potential, normalized to e/R, Eq. (17), this is singular when psi=0
       # so we set potential to 0 at origin
       # Note that psi in this formula should be obtained from the exact equation, not 1D ultra-relativsitic one.
       kernel[non_origin] = (beta*(1.0-beta*beta*np.cos(alpha[non_origin] + psi[non_origin])) / psi[non_origin] /(1.0-beta*np.sin(eta[non_origin]))) 
       kernel[origin] = 0.0
       field = np.transpose(field.T * kernel)

    if (method == "potential_regularized") :
       # regulaized 1D lw potential (phi-beta*A_s), normalized to e/R
       # note this converges at fewer (e.g. 1000) points than using full lw potential
       # but the subtracted term needs to be accounted for after convolution
       kernel = (-beta*(1.0-beta) + 
              1*beta*(1.0-beta*beta*np.cos(alpha + psi)) /(1.0-beta*np.sin(eta))) /psi
       field = np.transpose(field.T * kernel)    
    
    if (filter) :
        # select wavelets based on filter
        filtered_x = eval(filter_exp)  
        wx_filtered = wx[filtered_x]
        wy_filtered = wy[filtered_x]
        fld_filtered = field[filtered_x]
    else :
        wx_filtered = wx
        wy_filtered = wy
        fld_filtered = field
        
    #if (mpi_rank == 0) : print("fld min/max:", fld_filtered.min(), fld_filtered.max())

    ## unscale if necessary
    if unscale_coord :
        wx_filtered /= _gamma**3.0
        wy_filtered /= _gamma**2.0
    
    if (flatten) :
       fld1 = fld_filtered.flatten()
       flds = fld1
       if (num_fields>1) :
          flds = np.stack([fld1]*num_fields)
       return (wx_filtered.flatten(), wy_filtered.flatten(), flds)
    else :
       return wx, wy, field, psi, eta


def retard_angle_1D_ultra_relativistic(alpha_axis) :

    psi = np.zeros_like(alpha_axis)
    eta = np.zeros_like(alpha_axis)

    positive_axis = (alpha_axis>0)
    negative_axis = (alpha_axis<0)
    psi[positive_axis] = (24.0*alpha_axis[positive_axis])**(1.0/3.0)
    psi[negative_axis] = -0.5*alpha_axis[negative_axis]
    eta[positive_axis] = 0.5*(np.pi - alpha_axis[positive_axis] - psi[positive_axis]) 
    eta[negative_axis] = 0.5*(-np.pi - alpha_axis[negative_axis] - psi[negative_axis])

    return (psi, eta)


# general retarded angle solver (vectorized)
def retarded_angle(alpha=np.array([0.0]), beta=0.0, epsilon=0.0, y2=0.0):
    from scipy.optimize import fsolve

    psi = np.zeros_like(alpha)
    eta = np.zeros_like(alpha)

    positive_axis = (alpha>0)
    zero_point = (alpha==0)
    negative_axis = (alpha<0)

    # y2 is y**2
    #psi_max = (24.0*alpha.max())**(1.0/3.0)
    #x0s = np.ones_like(alpha)*psi_max*1.2
    
    slice_len = 200
    for x in range(0, len(alpha), slice_len) :
        alpha_slice = alpha[x:x+slice_len].copy()
        alpha_max = alpha_slice.max()
        psi_max = np.abs(alpha_max)
        if (alpha_max>0.0) : psi_max = (24.0*alpha_max)**(1.0/3.0)
        x0s_slice = np.ones_like(alpha_slice)*psi_max*1.0        
        arg_list = alpha_slice.tolist()
        arg_list.extend((beta, epsilon, y2))
        psi_slice = fsolve(func,x0=x0s_slice, args=arg_list, maxfev=10000)
        psi[x:x+slice_len] = psi_slice

    if (epsilon ==0.0 and y2==0.0) :
       psi[zero_point] = 0.0
    
    eta[positive_axis] = 0.5*(np.pi - alpha[positive_axis] - psi[positive_axis]) 
    eta[negative_axis] = 0.5*(-np.pi - alpha[negative_axis] - psi[negative_axis])
    eta[zero_point] = np.pi/2.0
    
    return (psi, eta)


def func(x, args_array) :
    # this is the function relating alpha and retarded angle psi
    # python version
    y2 = args_array[-1]
    epsilon = args_array[-2]
    beta = args_array[-3]
    alpha = args_array[:-3]
    #ret = y2 + 1.0+(1.0+epsilon)**2.0 - 2.0*(1.0+epsilon)*np.cos(alpha+x) - (x/beta)**2.0
    ret = y2 + 4.0*(1.0+epsilon)*(np.sin((alpha+x)/2.0))**2.0 + epsilon**2.0 - (x/beta)**2.0
    return ret.tolist()