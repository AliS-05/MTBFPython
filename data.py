import pandas as pd
import numpy as np
from csv import writer
from openpyxl import load_workbook
#constants
NUM_SUBSYSTEMS = 29
FAILURE_TYPES = 3 # 1, 2, 6
MAINTENANCE_DATA_FILEPATH = "./data/maintenanceDataReal.csv"
CONTRACTOR_MTBF_FILEPATH = "./data/predictedReal.csv"
OUTPUT_FILEPATH = "./web/static/graphs"
DATASET_FILEPATH = "/mnt/c/Users/sefra/Downloads/dataset.xlsx"

def datasetCleanMaintenanceData():
    maintenanceData = pd.read_excel(DATASET_FILEPATH, sheet_name="Failure Data")
    maintenanceData = maintenanceData.drop(maintenanceData.filter(regex='^Unnamed').columns, axis=1)
    return maintenanceData

def datasetCleanContractorData():
    contractorMTBF = pd.read_excel(DATASET_FILEPATH, sheet_name="Initial MTBF", skiprows=1)
    contractorMTBF = contractorMTBF.drop(contractorMTBF.filter(regex='^Unnamed').columns, axis=1)
    contractorMTBF["SubSystem"] = pd.to_numeric(contractorMTBF["SubSystem"], errors="coerce")
    contractorMTBF = contractorMTBF.dropna(subset=["SubSystem"])
    contractorMTBF["SubSystem"] = contractorMTBF["SubSystem"].astype(int)
    return contractorMTBF


def cleanMaintenanceData():
    maintenanceData = pd.read_csv(MAINTENANCE_DATA_FILEPATH)
    #remove all blank separator columns
    maintenanceData = maintenanceData.drop(maintenanceData.filter(regex='^Unnamed').columns, axis = 1)
    return maintenanceData


#skip first row "Failure Type,1,2,6"

def cleanContractorData():
    contractorMTBF = pd.read_csv(CONTRACTOR_MTBF_FILEPATH, skiprows=1)
    contractorMTBF = contractorMTBF.drop(contractorMTBF.filter(regex='^Unnamed').columns, axis=1)
    contractorMTBF["SubSystem"] = pd.to_numeric(contractorMTBF["SubSystem"], errors="coerce")

    contractorMTBF = contractorMTBF.dropna(subset=["SubSystem"])

    contractorMTBF["SubSystem"] = contractorMTBF["SubSystem"].astype(int)
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


def addEntryToData(date, hours, system, subSystem, failureType):
    fmtDate = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
    newRow = [fmtDate.strftime("%y/%m/%d"), hours, system, subSystem, failureType]
    with open(MAINTENANCE_DATA_FILEPATH, 'a', newline = '') as f:
        writerObj = writer(f)
        writerObj.writerow(newRow)

def removeEntry(date, system, subSystem, failureType):
       return
        
def undoEntry():
    with open(MAINTENANCE_DATA_FILEPATH, "rb+") as f:
        f.seek(0, 2)
        pos = f.tell() - 1

        while pos >= 0:
            f.seek(pos)
            if f.read(1) not in b"\r\n":
                break
            pos -= 1

        while pos >= 0:
            f.seek(pos)
            if f.read(1) == b"\n":
                break
            pos -= 1

        f.truncate(pos + 1)


