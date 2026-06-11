import pandas as pd #pd for data tables
import numpy as np #np for general data manipulation and number crunching
import scipy.stats as stats #stats for stats functions
import matplotlib.pyplot as plt #plt for plotting 
from collections import defaultdict #for not having to worry about initializing dicts
import os



#constants
NUM_SUBSYSTEMS = 29
FAILURE_TYPES = 3 # 1, 2, 6
MAINTENANCE_DATA_FILEPATH = "/mnt/c/Users/sefra/Downloads/maintenance dataset(Failure Data).csv"
CONTRACTOR_MTBF_FILEPATH = "/mnt/c/Users/sefra/Downloads/maintenance dataset(Initial MTBF).csv"
os.makedirs("graphs", exist_ok=True)

maintenanceData = pd.read_csv(MAINTENANCE_DATA_FILEPATH)
contractorMTBF = pd.read_csv(CONTRACTOR_MTBF_FILEPATH, skiprows=1)

#remove all blank separator columns
maintenanceData.drop(maintenanceData.filter(regex='^Unnamed').columns, axis = 1, inplace=True)


#type hinting tuple key and float value
#ie (subsystem, failureType) -> 1 + amount of failures
nStar : dict[tuple[int, int], float] = defaultdict(lambda: 1)

#same thing for tauStar
#(subsystem, failureType) -> flightHours
#initalizing with manufacturer estimates
tauStar : dict[tuple[int, int], float] = {}
for _, row in contractorMTBF.iterrows():
    tauStar[row["SubSystem"], 1] = round(row['MTBF Inherent (hrs)'], 3)
    tauStar[row["SubSystem"], 2] = round(row["MTBF Induced (hrs)"], 3)
    tauStar[row["SubSystem"], 6] = round(row["MTBF No Defect (hrs)"], 3)
print(tauStar)

flightHours = round(maintenanceData["Flight Hours"].sum(numeric_only=True), 3)

#stop before Factor Columns
factorCol = maintenanceData.columns.get_loc("Factor 1")

runningFlightHours = 0

thetaHat : dict[tuple[int, int], float] = {}

#dict with key (subsystem, failureType) and 
#value tuple (date, ourEstimatedMTBF, upperBound, lowerBound)
#defaultdict(list) so i dont have to worry about initialization
graphDict : dict[tuple[int,int], list[tuple[any, float, float, float]]] = defaultdict(list)

for num, row in maintenanceData.iterrows():
    runningFlightHours += row["Flight Hours"]

    subCol = row.iloc[3:factorCol:2]
    typeCol = row.iloc[4:factorCol:2]

    printed = False
    for sub, typ in zip(subCol, typeCol):
        if pd.notna(sub) and pd.notna(typ):
            typ = int(typ)
            print(sub, typ, end=" ")
            printed = True           
            #n_jk
            n = nStar.get((sub,typ), 1) + 1
            nStar[(sub,typ)] = n

            #tau_jk
            tau = tauStar.get((sub,typ), 1) + runningFlightHours

            #our estimate
            mtbfEstimate = round(tau / n, 3)
            thetaHat[(sub,typ)] = mtbfEstimate

            #upperBound
            confidenceIntervalUpper = 2*tau/(stats.chi2.ppf(0.025, df=2*n))

            #lowerBound
            confidenceIntervalLower = 2*tau/(stats.chi2.ppf(0.975, df=2*n))
            graphDict[(sub,typ)].append((row["Date"], mtbfEstimate, confidenceIntervalUpper, confidenceIntervalLower))
        else:
            continue
    if printed:
        print() 
        #continue

#i = (sub, type)
#thetaHat.get(i) = estimate
for i in tauStar:
    if not pd.notna(thetaHat.get(i)):
        thetaHat[i] = (runningFlightHours + tauStar[i]) / nStar[i]

thetaHatDf = pd.DataFrame([
    {"Subsystem": sub, "Failure Type": ft, "MTBF Estimate (hrs)": theta}
    for(sub, ft), theta in sorted(thetaHat.items())
])


table3 = thetaHatDf.pivot(index='Subsystem', columns='Failure Type', values='MTBF Estimate (hrs)')
table3.columns = ['Type 1 (Inherent)', 'Type 2 (Induced)', 'Type 6 (No Defect)']
table3.index.name = 'Subsystem'

print(table3.to_string())

contractorEstimate = {}
for _, row in contractorMTBF.iterrows():
    contractorEstimate[row["SubSystem"], 1] = row['MTBF Inherent (hrs)']
    contractorEstimate[row["SubSystem"], 2] = row['MTBF Induced (hrs)']
    contractorEstimate[row["SubSystem"], 6] = row['MTBF No Defect (hrs)']

for (sub, typ), series in graphDict.items():
    print((sub,typ))
    dates, theta, upper, lower = zip(*series)
    plt.figure(figsize=(10, 5))
    plt.axhline(contractorEstimate[(sub, typ)], color='blue', linestyle=':', label='Contractor MTBF')
    plt.plot(dates, theta, 'r-', label='Bayes Estimate')
    plt.plot(dates, upper, 'r--', label='95% CI')
    plt.plot(dates, lower, 'r--')
    plt.title(f"Subsystem {sub} Type {typ}")
    plt.xlabel("Date")
    plt.ylabel("MTBF (hrs)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"graphs/{sub}_{typ}.png")
    plt.close()
