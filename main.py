import pandas as pd #pd for data tables
import numpy as np #np for general data manipulation and number crunching
import scipy.stats as stats #stats for stats functions
import matplotlib.pyplot as plt #plt for plotting 

#constants
NUM_SUBSYSTEMS = 29
FAILURE_TYPES = 3 # 1, 2, 6
MAINTENANCE_DATA_FILEPATH = "/mnt/c/Users/sefra/Downloads/maintenance dataset(Failure Data).csv"
CONTRACTOR_MTBF_FILEPATH = "/mnt/c/Users/sefra/Downloads/maintenance dataset(Initial MTBF).csv"

maintenanceData = pd.read_csv("/mnt/c/Users/sefra/Downloads/maintenance dataset(Failure Data).csv")
#skip first row "Failure Type,1,2,6"
contractorMTBF = pd.read_csv("/mnt/c/Users/sefra/Downloads/maintenance dataset(Initial MTBF).csv", skiprows=1)

# reshape csv format to give each failure its own row for ease of access
sub_cols = ['Sub'] + [f'Sub.{i}' for i in range(1, 13)]
type_cols = ['Type'] + [f'Type.{i}' for i in range(1, 13)]

rows = []
for _, row in maintenanceData.iterrows():
    for sub_col, type_col in zip(sub_cols, type_cols):
        sub = row[sub_col]
        failure_type = row[type_col]
        if pd.notna(sub) and pd.notna(failure_type):
            rows.append({
                'date': row['Date'],
                'flight_hours': row['Flight Hours'],
                'system': row['System'],
                'subsystem': int(sub),
                'failure_type': int(failure_type)
            })

long_df = pd.DataFrame(rows)

failure_counts = long_df.groupby(['subsystem', 'failure_type']).size().reset_index(name='count')

nStar = {(row['subsystem'], row['failure_type']): 1 + row['count'] for _, row in failure_counts.iterrows()}


# calculate tauStar
#calculate total flight hours recorded in dataset first.
#then add that number to contractorMTBF and add result into tauStar

flightHours = round(maintenanceData["Flight Hours"].sum(numeric_only=True), 3)



#SubSystem, FailureType
#tauStar: dict[tuple[int, int], float] = {}

tauStar = {}
for _, row in contractorMTBF.iterrows():
    tauStar[row["SubSystem"], 1] = round(row['MTBF Inherent (hrs)'] + flightHours, 3)
    tauStar[row["SubSystem"], 2] = round(row["MTBF Induced (hrs)"] + flightHours, 3)
    tauStar[row["SubSystem"], 6] = round(row["MTBF No Defect (hrs)"] + flightHours, 3)



# nHatBayes = n* / tau* — Bayes estimator of failure rate
#nHatBayes: dict[tuple[int, int], float] = {}
nHatBayes = {}
for (subsystem, failureType), tau in tauStar.items():
    n = nStar.get((subsystem, failureType), 1) #returns 1 by default
    # n*_jk / tau*_jk
    nHatBayes[subsystem, failureType] = round(n / tau, 6)

# theta_hat = tau* / n* — Bayes MTBF estimate (Table 3)
#thetaHat is just inverse of nHatBayes, table 3 output

#thetaHat: dict[tuple[int, int], float] = {}
thetaHat = {}
for (subsystem, failureType), tau in tauStar.items():
    n = nStar.get((subsystem, failureType), 1)
    thetaHat[subsystem, failureType] = round(tau / n, 3)

thetaHatDf = pd.DataFrame([
    {"Subsystem": sub, "Failure Type": ft, "MTBF Estimate (hrs)": theta}
    for(sub, ft), theta in sorted(thetaHat.items())
])


table3 = thetaHatDf.pivot(index='Subsystem', columns='Failure Type', values='MTBF Estimate (hrs)')
table3.columns = ['Type 1 (Inherent)', 'Type 2 (Induced)', 'Type 6 (No Defect)']
table3.index.name = 'Subsystem'

print(table3.to_string())



confidenceIntervalUpper = {}
confidenceIntervalLower = {}

for (subsystem, failureType), tau in tauStar.items():
    n = nStar.get((subsystem, failureType), 1)
    #equi-tailed two sided credible interval with 2n degrees of freedom
    confidenceIntervalUpper[(subsystem, failureType)] = 2*tau/(stats.chi2.ppf(0.025, df=2*n))
    confidenceIntervalLower[(subsystem, failureType)] = 2*tau/(stats.chi2.ppf(0.975, df=2*n))

for i in confidenceIntervalUpper, confidenceIntervalLower:
    print(i)


