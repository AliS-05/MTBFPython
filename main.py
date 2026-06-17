import pandas as pd #pd for data tables
import numpy as np #np for general data manipulation and number crunching
import scipy.stats as stats #stats for stats functions
import scipy.special as sp #gammainc, factorial
import matplotlib.pyplot as plt #plt for plotting 
import data


maintenanceData = data.cleanMaintenanceData()
flightHours = data.calculateTotalFlightHours(maintenanceData)

maintenanceData = data.reshapeMaintenanceData(maintenanceData)

contractorMTBF = data.cleanContractorData()

contractorEstimates = data.constructContractorEstimates(contractorMTBF)

nStar : dict[tuple[int,int], float] = {}
tauStar: dict[tuple[int, int], float] = {}
thetaHat: dict[tuple[int, int], float] = {}
bayesFactor: dict[tuple[int, int], float] = {}

def calculateNStar():
    failureCounts = maintenanceData.groupby(['subsystem', 'failure_type']).size().reset_index(name='count')
    for _, row in failureCounts.iterrows():
        nStar[(row['subsystem'], row['failure_type'])] =  1 + row['count']

def calculateTauStar():
    for _, row in contractorMTBF.iterrows():
        tauStar[row["SubSystem"], 1] = round(row["MTBF Inherent (hrs)"] + flightHours, 3)
        tauStar[row["SubSystem"], 2] = round(row["MTBF Induced (hrs)"] + flightHours, 3)
        tauStar[row["SubSystem"], 6] = round(row["MTBF No Defect (hrs)"] + flightHours, 3)

def calculateBayesEstimate():
    # theta_hat = tau* / n* 
    for (subsystem, failureType), tau in tauStar.items():
        n = nStar.get((subsystem, failureType), 1)
        thetaHat[(subsystem, failureType)] = round(tau / n, 1)

    thetaHatDf = pd.DataFrame([
        {"Subsystem": sub, "Failure Type": ft, "MTBF Estimate (hrs)": theta}
        for(sub, ft), theta in sorted(thetaHat.items())
    ])
    #printable table
    table3 = thetaHatDf.pivot(index='Subsystem', columns='Failure Type', values='MTBF Estimate (hrs)')
    table3.columns = ['Type 1 (Inherent)', 'Type 2 (Induced)', 'Type 6 (No Defect)']
    table3.index.name = 'Subsystem'
    
    return table3

def calculateBayesFactor():
    #calculating bayes factor
    beta = 0.1
    prior = np.exp(-1/(1-beta)) / (1 - np.exp(-1/(1-beta)))  # P(H1)/P(H0), constant for fixed beta

    for (sub, typ), tau in tauStar.items():
        theta0 = contractorEstimates[(sub, typ)]
        failures = nStar.get((sub,typ), 1)
        x = tau / ((1-beta) * theta0)
        #lower incomplete
        probH0 = sp.gammainc(nStar.get((sub,typ),1), (tau / ((1-beta) * theta0)))
        #upper incomplete 'incc'
        probH1 = sp.gammaincc(nStar.get((sub,typ),1), (tau / ((1-beta) * theta0))) 
        bayesFactor[(sub, typ)] = round((probH1 / probH0) / prior, 1)

    bfDf = pd.DataFrame([
      {"Subsystem": sub, "Failure Type": ft, "BF": bf}
      for (sub, ft), bf in sorted(bayesFactor.items())
    ])
    
    table6 = bfDf.pivot(index='Subsystem', columns='Failure Type', values='BF')
    table6.columns = ['BF for Type 1', 'BF for Type 2', 'BF for Type 6']
    table6.index.name = 'Subsystem'

    #formatting so table doesnt pritn every value scientifically
    def fmt(x):
        if pd.isna(x):
            return "NaN"
        if abs(x) > 1e6:
            return f"{x:.1e}"
        return f"{x:.1f}"

    return table6.to_html(float_format=fmt)

def returnBayesEstimates():
    maintenanceData = data.cleanMaintenanceData()
    flightHours = data.calculateTotalFlightHours(maintenanceData)
    maintenanceData = data.reshapeMaintenanceData(maintenanceData)
    contractorMTBF = data.cleanContractorData()
    contractorEstimates = data.constructContractorEstimates(contractorMTBF)

    calculateNStar()
    calculateTauStar()
    return calculateBayesEstimate()


def returnBayesFactor():
    maintenanceData = data.cleanMaintenanceData()
    flightHours = data.calculateTotalFlightHours(maintenanceData)
    maintenanceData = data.reshapeMaintenanceData(maintenanceData)
    contractorMTBF = data.cleanContractorData()
    contractorEstimates = data.constructContractorEstimates(contractorMTBF)


    calculateNStar()
    calculateTauStar()
    return calculateBayesFactor()

def findWorstPerformingSubSystems():
    #return sorted ratios, caller can use splicing to get what they want
    systemRatios = []
    for (subsystem, failureType), estimate in thetaHat.items():
        ratio = estimate / contractorEstimates[(subsystem, failureType)]
        systemRatios.append(((int(subsystem), failureType), round(ratio,2)))
    #sorts based on estimate, ie index 1 of tuple   
    return sorted(systemRatios, key = lambda x: x[1])[0:5]

def main():
    calculateNStar()
    calculateTauStar()
    res.append(calculateBayesEstimate())
    res.append(calculateBayesFactor())
    return res

if __name__ == "__main__":
    main()


