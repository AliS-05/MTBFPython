import pandas as pd
import numpy as np

#constants
NUM_SUBSYSTEMS = 29
FAILURE_TYPES = 3 # 1, 2, 6
MAINTENANCE_DATA_FILEPATH = "/mnt/c/Users/sefra/Downloads/maintenanceDataReal.csv"
CONTRACTOR_MTBF_FILEPATH = "/mnt/c/Users/sefra/Downloads/predictedReal.csv"
OUTPUT_FILEPATH = "./graphs"

def cleanMaintenanceData():
    maintenanceData = pd.read_csv(MAINTENANCE_DATA_FILEPATH)
    #remove all blank separator columns
    maintenanceData = maintenanceData.drop(maintenanceData.filter(regex='^Unnamed').columns, axis = 1)
    return maintenanceData


#skip first row "Failure Type,1,2,6"

def cleanContractorData():
    contractorMTBF = pd.read_csv(CONTRACTOR_MTBF_FILEPATH, skiprows=1)
    contractorMTBF["SubSystem"] = pd.to_numeric(contractorMTBF["SubSystem"], errors="coerce")
    contractorMTBF = contractorMTBF.dropna(subset=["SubSystem"])
    return contractorMTBF

def constructContractorEstimates(contractorEstimatesDataFrame) -> dict[tuple[int, int], float]:
    contractorEstimates : dict[tuple[int, int], float] = {}
    for _, row in contractorEstimatesDataFrame.iterrows():
        contractorEstimates[row["SubSystem"], 1] = round(row['MTBF Inherent (hrs)'], 3)
        contractorEstimates[row["SubSystem"], 2] = round(row["MTBF Induced (hrs)"], 3)
        contractorEstimates[row["SubSystem"], 6] = round(row["MTBF No Defect (hrs)"], 3)
    return contractorEstimates

def reshapeMaintenanceData(maintenanceDataDataFrame):
    # reshape csv format to give each failure its own row for ease of access
    sub_cols = ['Sub'] + [f'Sub.{i}' for i in range(1, 13)]
    type_cols = ['Type'] + [f'Type.{i}' for i in range(1, 13)]

    rows = []
    for _, row in maintenanceDataDataFrame.iterrows():
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

    return pd.DataFrame(rows)

def calculateTotalFlightHours(maintenanceData):
    return round(maintenanceData["Flight Hours"].sum(numeric_only=True), 3)
