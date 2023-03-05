import numpy as np

from scipy import spatial

from scipy.optimize      import nnls

from math import acos, degrees

def get_q(lambda0,refractiveIndex,scatteringAngle):
    
    # Calculate the Bragg wave vector
    # scatteringAngle in radians (2pi rad equals 360 degrees).
    # labmda in nanometers
    
    lambda0 = lambda0 / 1e9 # To meters
    
    # Output units are 1/m
    
    try: 
        refractiveIndex =  complex(refractiveIndex).real
    except:
        return None

    return 4*np.pi*refractiveIndex/lambda0 * np.sin(scatteringAngle/2)

def hydrodynamic_radius(D,temp,viscosity):
    
    # D is the single-particle diffusion coefficient
    # D units are m**2 / seconds
    # viscosity units are pascal-second (Newton-second per square meter)
    
    # Returns the Stokes–Einstein radius
    
    kb    = 1.380649e-23 # Joul / Kelvin
    
    return kb*temp / (6*np.pi*viscosity*D) # Units are meters

def diffusion_from_hydrodynamic_radius(hr,temp,viscosity):

    # Compute the single-particle diffusion coefficient (units are m**2 / seconds)
    # viscosity units are pascal-second (Newton-second per square meter)
    # temp in kelvin

    kb    = 1.380649e-23 # Joul / Kelvin

    return kb*temp / (6*np.pi*viscosity*hr) # Units are meters

def diffusion_from_inverse_decay_rate(s,q):
    
    # s is the inverse of the decay rate (gamma)
    # q is derived from get_q()

    return 1 / (s*(q**2))

def s_inverse_decay_rate(D,q):
    
    # D is the single-particle diffusion coefficient
    # D units are m**2 / seconds
    
    return 1 / (D*(q**2)) # units are seconds

def g1_first_order_corr(s,t):
    
    # s is the inverse of the decay rate (gamma)
    
    gamma = 1/s
    
    return np.exp(-gamma*t) # unitless
    
def g2_second_order_corr(g1,beta):
    
    # g1 is the first order autocorrelation

    return 1 + beta * (g1)**2 # unitless

def g1_from_g2(g2,beta):
        
    # g2 is the second order autocorrelation

    return np.sqrt( (g2-1) / beta) # unitless

def get_beta_prior(g2,time):

    """ 
    Requires -  
                g2 matrix n*m
                Time vector of length n

    Returns the intercept estimate 
    """


    npFit     = np.polyfit( time[time < 5*1e-6],np.log(g2[time < 5*1e-6,:] - 1), 2 )
    betaPrior = np.exp(npFit[-1])
        
    return betaPrior

def tikhonov_Phillips_reg(kernel,alpha,data,W):

    """
    
    Solve x / minimize ||w(Ax - b)|| + alpha||Mx|| 

    A is the kernel
    b is the vector with the measurements
    x is the unknown vector we want to estimate
    W is the weights vectors
    M is the second order derivative matrix

    Moreover, we add a last equation to the system such that sum(x) equals 1!
                and we force the initial and last values to be equal to 0!

    This function is was based on a 
    Quora answer (https://scicomp.stackexchange.com/questions/10671/tikhonov-regularization-in-the-non-negative-least-square-nnls-pythonscipy) 
    given by Dr. Brian Borchers (https://scicomp.stackexchange.com/users/2150/brian-borchers)

    """

    W      = np.append(W,np.array([1e3,1e3,1e3])) # weight to force the initial and last values equal to 0, and the sum of contributions equal to 1

    rowToForceInitialValue    = np.zeros(kernel.shape[1])
    rowToForceInitialValue[0] = 1
    rowToForceLastValue       = np.flip(rowToForceInitialValue)

    data   = np.sqrt(W)*np.append(data,np.array([1,0,0])) 
    data   = data.reshape(-1, 1)

    kernel = np.vstack([kernel,np.ones(kernel.shape[1]),rowToForceInitialValue,rowToForceLastValue])
    kernel = np.sqrt(W)[:, None] * kernel # Fidelity term
    
    cols   = kernel.shape[1]
    
    M = np.zeros((cols,cols))
    for i in range(1,M.shape[1]-1):
        M[i,i-1] = -1
        M[i,i]   =  2
        M[i,i+1] = -1
    
    L      = np.sqrt(alpha) * M                     # Penalty term
    C      = np.concatenate([kernel, L], axis=0)    
    d      = np.concatenate([data, np.zeros(cols).reshape(-1, 1)])
    
    # residual from || Ax-b ||_2
    x, residualNorm   = nnls(C, d.flatten())

    # Get the norm of the penalty term
    penaltyNorm = np.linalg.norm(M.dot(x),2)

    return x, residualNorm, penaltyNorm

def get_contributios_prior(g1_autocorrelation,time,s_space,betaPrior,alpha,weights=None):

    """
        Input -
                
            g1 autocorrelation matrix n-points m-datasets
            time vector of length n
            s_space to create the kernel for the Thinkohonov regularization function
            betaPrior vector of length m

        Returns -
            
            The estimated contribution of each decay rate (length defined by the s_space vector)

    """

    nDatasets = g1_autocorrelation.shape[1]

    # Convert to list if required - we need one reg term (aka alpha) per curve
    if type(alpha) is not list:

        alphaList = [alpha for _ in range(nDatasets)]
    else:
        # No need to convert to list
        alphaList = alpha

    s      = s_space.reshape((-1, 1))

    sM, tM = np.meshgrid(s, time, indexing='xy')
    A      = np.exp(-tM/sM)

    contributions = []
    residuals     = []
    penaltyNorms  = []

    for i in range(nDatasets):

        g1temp     = g1_autocorrelation[:,i]

        try:
            maxID      = np.min(np.argwhere(np.isnan(g1temp)))
        except:
            maxID      = len(g1temp)

        try:

            if weights is  None:
                weightsIdx = np.arange(maxID)*0 + 1 # Equal weights
            else:
                weightsIdx = weights[:maxID,i] # Custom weights

            g1Filtered = g1temp[:maxID]
            Afiltered  = A[:maxID,:]

            cont, residual, penaltyNorm   = tikhonov_Phillips_reg(Afiltered,alphaList[i],g1Filtered,weightsIdx)

        # If the fitting didn't work!
        except:

            cont        = [0]
            residuals   = [0]
            penaltyNorm = [0]

        contributions.append(np.array(cont))
        residuals.append(residual)
        penaltyNorms.append(penaltyNorm)

    return contributions, residuals, penaltyNorms

def g2_finite_aproximation(decay_rates,times,beta,contributions):
              
    """
    
    Obtain the autocorrelation function based on decay rates and their
    relatives contributions

    beta is the intercept (g2 at time 0)

    """

    assert len(decay_rates)   == len(contributions)
    
    g1 = np.array([np.sum(contributions*np.exp(-decay_rates*t)) for t in times])
            
    return 1 + beta * (g1)**2

def cosLawAngle(d1,d2,d3):
    
    """
    Use the three distances between vertices to get the angle 

    d1 = AB segment
    d2 = AC segment
    d3 = BC segment
    """

    return (degrees(acos((d1**2 + d2**2 - d3**2)/(2.0 * d1 * d2)))) 

def find_Lcurve_corner(residualsNorm,contributionsNorm):
 
    """
    Use the triangle method to find the corner of the L-curve

    If you use this function please cite the original manuscript from which I took the idea: 
                    Castellanos, J. Longina, Susana Gómez, and Valia Guerra. 
                    "The triangle method for finding the corner of the L-curve." 
                    Applied Numerical Mathematics 43.4 (2002): 359-373.

    Input - the norm vector of the residuals and the norm vector of the contributions
            i.e., the norm of the fidelity term and the norm of the penalty term

    Returns the position of the corner of the curve log(contributionsNorm) vs log(residualsNorm) 
    """

    # Convert to log
    x = np.log(residualsNorm)
    y = np.log(contributionsNorm)

    # Normalise to avoid floating point errors - This doesn't change the shape of the curve
    x = (x - np.min(x)) / (np.max(x)-np.min(x)) * 100
    y = (y - np.min(y)) / (np.max(y)-np.min(y)) * 100

    nPoints = len(x)

    angles   =  []
    poi2     =  []

    C = (x[nPoints-1],y[nPoints-1]) # Last point of the curve
    for i in range(nPoints-4):
        B = (x[i],y[i])
        d3 = spatial.distance.cdist([B],[C])[0]

        for ii in range(i+2,nPoints-2):
            A  = (x[ii],y[ii])   
            d1 = spatial.distance.cdist([B],[A])[0]
            d2 = spatial.distance.cdist([A],[C])[0]

            area = (B[0] - A[0])*(A[1]-C[1]) - (A[0] - C[0])*(B[1]-A[1])
            
            angle = cosLawAngle(d1,d2,d3)
            
            if angle < 160 and area > 0:
                
                poi2.append(ii)
                angles.append(angle)
       
    try:

        selIdx2    = poi2[np.argmin(angles)]
        return selIdx2 

    except:

        return None