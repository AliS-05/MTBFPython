import pandas as pd


#constants
NUM_SUBSYSTEMS = 29
FAILURE_TYPES = 3 # 1, 2, 6
MAINTENANCE_DATA_FILEPATH = "/mnt/c/Users/sefra/Downloads/maintenanceDataReal.csv"
CONTRACTOR_MTBF_FILEPATH = "/mnt/c/Users/sefra/Downloads/predictedReal.csv"
OUTPUT_FILEPATH = "./graphs"

def cleanMaintenanceData(MAINTENANCE_DATA_FILEPATH):
    maintenanceData = pd.read_csv(MAINTENANCE_DATA_FILEPATH)
    #remove all blank separator columns
    maintenanceData.drop(maintenanceData.filter(regex='^Unnamed').columns, axis = 1, inplace=True)

#skip first row "Failure Type,1,2,6"
contractorMTBF = pd.read_csv(CONTRACTOR_MTBF_FILEPATH, skiprows=1)
contractorMTBF["SubSystem"] = pd.to_numeric(contractorMTBF["SubSystem"], errors="coerce")
contractorMTBF = contractorMTBF.dropna(subset=["SubSystem"])


