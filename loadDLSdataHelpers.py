import pandas as pd 
import numpy  as np 

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def isBlank(myString):

    return (not (myString and myString.strip())) or myString == "nan"

def readWyatFile(file):

    """

    Input: CSV  where the first column stores the time in microseconds
    all subsequent columns store the measured autocorrelation

    """

    df = pd.read_csv(file, sep=',',encoding=' latin-1')

    df = df.sort_values(by=[df.columns[0]], ascending=True)

    # Delete the first point 
    startIDx        = 1

    time            = np.array(df.iloc[startIDx:,0]) / 1e6 # From microseconds to seconds 
    total_rows      = len(df.index)
    
    # Get columns with data (not too many NaNs)
    selInds     = []
    sampleNames = []

    for ind, column in enumerate(df.columns[1:]):

        nas = df[column].isna().sum()
        if nas < total_rows / 2:
            selInds.append(ind)
            sampleNames.append(column)

            # add 1 if required to the autocorrelation curves
            if np.min(np.abs(df[column])) < 0.01:
                df[column] += 1 

    autocorrelation = np.array(df.iloc[startIDx:,1:])[:,selInds]

    return time, autocorrelation, sampleNames