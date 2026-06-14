import pandas as pd #pd for data tables
import numpy as np #np for general data manipulation and number crunching
import scipy.stats as stats #stats for stats functions
import matplotlib.pyplot as plt #plt for plotting 
from collections import defaultdict #for not having to worry about initializing dicts
import os



#constants
NUM_SUBSYSTEMS = 29
FAILURE_TYPES = 3 # 1, 2, 6
MAINTENANCE_DATA_FILEPATH = "/mnt/c/Users/sefra/Downloads/maintenanceDataReal.csv"
CONTRACTOR_MTBF_FILEPATH = "/mnt/c/Users/sefra/Downloads/predictedReal.csv"
OUTPUT_FILEPATH = "./graphs"

maintenanceData = pd.read_csv(MAINTENANCE_DATA_FILEPATH)
#skip first row "Failure Type,1,2,6"
contractorMTBF = pd.read_csv(CONTRACTOR_MTBF_FILEPATH, skiprows=1)
contractorMTBF["SubSystem"] = pd.to_numeric(contractorMTBF["SubSystem"], errors="coerce")
contractorMTBF = contractorMTBF.dropna(subset=["SubSystem"])
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

for key, tau in list(tauStar.items()):
    if pd.isna(tau):
        del tauStar[key]

runningFlightHours = 0

thetaHat : dict[tuple[int, int], float] = {}

#dict with key (subsystem, failureType) and 
#value tuple (date, ourEstimatedMTBF, upperBound, lowerBound)
#defaultdict(list) so i dont have to worry about initialization
graphDict : dict[tuple[int,int], list[tuple[any, float, float, float]]] = defaultdict(list)


for num, row in maintenanceData.iterrows():
    runningFlightHours += row["Flight Hours"]
    subCol = row.iloc[3::2]
    typeCol = row.iloc[4::2]
    
    #iterate through sub,type pairs to update nStar for that date
    for sub,typ in zip(subCol, typeCol):
        if pd.notna(sub) and pd.notna(typ):
            typ = int(typ)
            #debug print statement, keeping in case i need it later
            #print(sub, typ, end=" ")
            printed = True           
            #n_jk 
            n = nStar.get((sub,typ), 1) + 1
            nStar[(sub,typ)] = n
    #nStar should now be up-to-date

    for (subsystem, failureType), tau in tauStar.items():
        n = nStar.get((subsystem, failureType), 1)  
        
        mtbfEstimate = round((tau + runningFlightHours) / n, 3)
        thetaHat[(subsystem, failureType)] = mtbfEstimate

        #upperBound
        confidenceIntervalUpper = 2*(tau + runningFlightHours)/(stats.chi2.ppf(0.025, df=2*n))

        #lowerBound
        confidenceIntervalLower = 2*(tau + runningFlightHours)/(stats.chi2.ppf(0.975, df=2*n))
        graphDict[(subsystem,failureType)].append((row["Date"], mtbfEstimate, confidenceIntervalUpper, confidenceIntervalLower))

for (sub, typ), series in graphDict.items():
      dates, theta, upper, lower = zip(*series)
      dates = pd.to_datetime(dates, format="%y/%m/%d")

      order = dates.argsort()
      dates = dates[order]
      theta = np.array(theta)[order]
      upper = np.array(upper)[order]
      lower = np.array(lower)[order]

      plt.figure(figsize=(20, 10))
      #str(int()) to prevent key errors just go with it
      plt.axhline(tauStar[(int(sub), typ)], color='blue', linestyle=':', label='Contractor MTBF')
      plt.plot(dates, theta, 'r-', label='Bayes Estimate')
      plt.plot(dates, upper, 'r--', label='95% CI')
      plt.plot(dates, lower, 'r--')

      yvals = list(theta) + list(lower) + [tauStar[(int(sub), typ)]]
      ymin, ymax = min(yvals), max(yvals)
      pad = (ymax - ymin) * 0.1
      plt.ylim(ymin - pad, ymax + pad)

      plt.title(f"Subsystem {sub} Type {typ}")
      plt.xlabel("Date")
      plt.ylabel("MTBF (hrs)")
      plt.legend()
      plt.grid(alpha=0.3)
      plt.xticks(rotation=45)
      plt.tight_layout()

      if not os.path.exists(OUTPUT_FILEPATH):
          os.makedirs(OUTPUT_FILEPATH)

      plt.savefig(f"{OUTPUT_FILEPATH}/{sub}_{typ}.png")

      plt.close()
