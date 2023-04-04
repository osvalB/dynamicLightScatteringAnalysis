import numpy as np
import pandas as pd

from helpers            import *
from loadDLSdataHelpers import *

class dls_experiment:

    """
    Class for the analysis of dynamic light scattering data
    This class was written by Osvaldo Burastero 
    
    No warranty whatsoever
    If you have questions please contact me:
    	oburastero@gmail.com

    Last time updated - Feb 2023

    """

    def __init__(self):

        # Experimental conditions
        self.temperature      = 293     # in K (20 degrees celsius)
        self.viscosity        = 8.9e-4  # pascal-second, default value of water
        self.refractiveIndex  = 1.33    # default value of water

        # Experimental setup
        self.lambda0, self.scatteringAngle  = None, None    # in nm & degrees
        self.scans, self.reads              = 1,1
        self.nMeasurements                  = None
        
        # Experimental results
        self.autocorrelation, self.time     = None, None
        
        # Metadata (to be filled with pandas dataframes)
        self.sampleInfo, self.sampleInfoRelevant  = None, None   

        # Derived quantity from the experimental setup
        self.q = None

        # Derived quantity from the results 
        self.g1 = None

        # Parameters required for fitting
        self.s_space, self.ds, self.hrs, self.weights     = None, None, None, None

        # Parameters required for fitting (L curve criteria)
        self.alphaVec, self.optimalAlpha, self.alphaOptIdx  = None, None, None

        # Fitted data
        self.betaGuess, self.contributionsGuess         = None, None  
        self.curvesResidualNorm, self.curvesPenaltyNorm = None, None  
        self.residualsG1                                = None

        # Predicted data
        self.autocorrelationPredicted = None

        return None

    def loadWyatFile(self,file):

        """
        Load Wyat plate reader output file
        1st column         - Time in microseconds
        2nd to End columns - Autocorrelation
        Headers            - Sample names chosen by the user
        """

        self.time, self.autocorrelationOriginal, sampleNames = readWyatFile(file)
        
        self.sampleInfo = pd.DataFrame({"conditions":sampleNames,"read":1,"scan":1,"include":True})

        self.lambda0         = 817 # in nm
        self.scatteringAngle = 150 / 180 * np.pi # radians

        return None

    def setAutocorrelationData(self):

        """

        Using the information from self.sampleInfo, which can be modified by the user,
        subset the autocorrelation data and retrieve the samples metadata we want to analyze

        """

        # Retrieve only the relevant column
        self.autocorrelation    = self.autocorrelationOriginal[:,self.sampleInfo.include]
        # Retrieve only the relevant rows
        self.sampleInfoRelevant = self.sampleInfo[self.sampleInfo.include]
        # Set the total number of samples to be analyzed
        self.nMeasurements      = self.autocorrelation.shape[1]

        return None

    def getQ(self):

        """
        Compute the Bragg wave vector
        """

        self.q = get_q(self.lambda0,self.refractiveIndex,self.scatteringAngle)

        return None

    def createFittingS_space(self,lowHr,highHr,n):

        """

        Create the s (inverse of gamma decay rate) space that will be used for the fitting
        The limits are given by the minimum and maximum desired hydrodynamic radius (in nanometers)
        
        Run after getQ()!

        """
        n = int(n) # Convert n to integer type for the np.logspace function


        sUpLimitHigh  = s_inverse_decay_rate(
            diffusion_from_hydrodynamic_radius(highHr/1e9,self.temperature,self.viscosity), self.q)

        sUpLimitLow   = s_inverse_decay_rate(
            diffusion_from_hydrodynamic_radius(lowHr/1e9,self.temperature,self.viscosity), self.q)

        # Sequence in linear space! 10.0**start to 10**stop
        self.s_space     = np.logspace(np.log10(sUpLimitLow),np.log10(sUpLimitHigh), n) 

        self.ds          = diffusion_from_inverse_decay_rate(self.s_space,self.q)
        self.hrs         = hydrodynamic_radius(self.ds ,self.temperature,self.viscosity)*1e9  # In nanometers

        return None

    def getBetaEstimate(self):

        """
        Fit a polynomial of degree 2 to the first 5 microseconds of data
        """

        self.betaGuess               = get_beta_prior(self.autocorrelation,self.time) 

        return None

    def getG1correlation(self):

        """ 
        Calculate the first order autocorrelation function g1
        """
        
        g1              = [g1_from_g2(self.autocorrelation[:,i],self.betaGuess[i]) for i in range(self.nMeasurements)]
        self.g1         = np.column_stack((g1))

        return None

    def getInitialEstimates(self,alpha=0.1,timeLimit=1e8):

        """

        Obtain initial estimates for the relative contributions

        Run after createFittingS_space() !

        timeLimit should be given in microseconds! Default time is 100 seconds (all the autocorrelation curve).

        alpha can be one value (same for all curves) or a list of values (one value per curve)
    
        """
 
        selectedTimes = self.time < (timeLimit / 1e6)

        # Return the fitted contributions and residuals of the first order autocorrelation function
        self.contributionsGuess, self.residualsG1, _   = get_contributios_prior(
            self.g1[selectedTimes,:],self.time[selectedTimes],self.s_space,self.betaGuess,alpha) 

        return None

    def getInitialEstimatesManyAlpha(self,
        alphaVec=(5**np.arange(-6,2,0.1,dtype=float))**2,timeLimit=1e8):

        """
        Apply the Tikhonov Philips regularisation for a given set of different values of alpha
        Useful to get afterwards the optimal alpha according to the L-curve criteria

        Result:

            We add curvesResidualNorm, curvesPenaltyNorm & alphaVec to the class object

            curvesResidualNorm contains the norm of the fidelity     term 
            curvesPenaltyNorm  contains the norm of the penalization term 
            alphaVec           contains the explored values of alpha

        """

        selectedTimes = self.time < (timeLimit / 1e6)

        curvesResidualNorm, curvesPenaltyNorm = [],[]

        self.alphaVec           = alphaVec

        # Iterate over the vector with different values of alpha
        for alpha in alphaVec:

            _ , residualNorm, penaltyNorm = get_contributios_prior(
                self.g1[selectedTimes,:],self.time[selectedTimes],
                self.s_space,self.betaGuess,alpha) 
          
            curvesResidualNorm.append(residualNorm) # List (one element per alpha) of lists (one element per curve)
            curvesPenaltyNorm.append(penaltyNorm)   # List (one element per alpha) of lists (one element per curve)

        self.curvesResidualNorm = np.array(curvesResidualNorm) # One row per alpha, one column per curve
        self.curvesPenaltyNorm  = np.array(curvesPenaltyNorm)  # One row per alpha, one column per curve

        return None

    def getOptimalAlphaLcurve(self):

        """
        Apply the triangle method to find the corner of the L-curve criteria en return the 'optimal' alpha for each curve
        """

        alphaOptIdx = []

        # Iterate over the curves
        for idx in range(self.curvesResidualNorm.shape[1]):

            alphaOptIdx.append(find_Lcurve_corner(self.curvesResidualNorm[:,idx],self.curvesPenaltyNorm[:,idx]))

        self.alphaOptIdx = alphaOptIdx

        return None

    def getInitialEstimatesOptimalAlphaLcurve(self,timeLimit=1e8):

        """
        Use the 'optimal' alpha selected using the L-curve corner criteria and the triangle method
        to estimate the distribution of (inverse) decay rates
        """

        self.optimalAlpha = [self.alphaVec[idx] for idx in self.alphaOptIdx]

        self.getInitialEstimates(self.optimalAlpha,timeLimit)

        return None

    def predictAutocorrelationCurves(self):

        # Create list to store the predicted autocorrelation data
        
        self.autocorrelationPredicted    = []

        for idx in range(self.nMeasurements):
            
            betaEst = self.betaGuess[idx]
            contEst = self.contributionsGuess[idx]

            # check that we estimated the contributions!
            if len(contEst) > 1:
                
                autocorrelationPredicted           =  g2_finite_aproximation(1 / self.s_space,self.time,betaEst,contEst)
                self.autocorrelationPredicted.append(np.array(autocorrelationPredicted))

            else:
                # In the case we couldn't fit anything!
                self.autocorrelationPredicted.append(np.array(0))

        self.autocorrelationPredicted = np.column_stack((self.autocorrelationPredicted))

        return None

    def getWeights(self):

        """
        Compare the fitted and experimental autocorrelation curve to get the residuals
        and assign weights to each point

        Caution: Not used in the Raynals online tool!
        """

        residuals      = np.subtract(self.autocorrelation,self.autocorrelationPredicted)
        weights        = 1 / np.abs(residuals)
        self.weights   = weights / weights.max(axis=0)

        return None

    def getWeightedInitialEstimates(self,alpha=0.15,timeLimit=1e8):

        """

        Call after fitting the g2 correlation curves
        that is, after running self.predictAutocorrelationCurves()

        Caution: Not used in the Raynals online tool!
        """

        if self.weights is None:

            self.getWeights()

        selectedTimes = self.time < (timeLimit / 1e6)

        # Return the fitted contributions and residuals of the first order autocorrelation function
        self.contributionsGuess, self.residualsG1   = get_contributios_prior(
            self.g1[selectedTimes,:],self.time[selectedTimes],self.s_space,self.betaGuess,alpha,self.weights) 

        return None

class dlsAnalyzer:

    """
    Useful to work with many different dls experiments
    """

    def __init__(self):

        """
        Create dictionary where each key value pair corresponds to 
            one DLS experiment, e.g. 
        """

        self.experimentsOri       = {}  
        self.experimentsModif     = {}  
        self.experimentNames      = []

        return None

    def loadExperiment(self,file,name):

        """
        Append one experiment to experimentsOri 
        """

        if name in self.experimentNames:

            return "Experiment name already selected!"

        try:

            self.experimentsOri[name] = dls_experiment()
            self.experimentsOri[name].loadWyatFile(file)
            self.experimentNames.append(name)

            return "Data loaded successfully!!!"

        except:

            pass
        
        return "Data could not be loaded"

    def deleteExperiment(self,name):

        self.experimentNames.remove(name)
        del self.experimentsOri[name]

        try: 
            del self.experimentsModif[name]
        except:
            pass

        return None

    def setExperimentProperties(self,experimentName,variable,value):

        """
        experimentName must be in self.experimentNames
        variable can be 'replicates', 'reads', or 'scans'
        value is a number
        """

        setattr(self.experimentsOri[experimentName], variable, value)

        return None

    def getExperimentProperties(self,variable):

        """
        variable can be 'replicates', 'reads', or 'scans'
        """

        return [getattr(self.experimentsOri[experimentName], variable) for experimentName in self.experimentNames]

if __name__ == "__main__":

    dls = dlsAnalyzer()
    l = dls.loadExperiment("test.csv","test")
    d = dls.experimentsOri["test"]
    d.lambda0 = 817
    d.scatteringAngle = 150 / 180 * np.pi
    d.getQ()
    d.createFittingS_space(0.09,1e6,200)
    d.setAutocorrelationData()
    d.getBetaEstimate()
    d.getG1correlation()
    d.getInitialEstimates()
    d.getInitialEstimatesManyAlpha()
    d.getOptimalAlphaLcurve()
    d.getInitialEstimatesOptimalAlphaLcurve()
    d.getInitialEstimatesManyAlpha()
