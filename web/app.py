import os
import pandas as pd
from flask import Flask, render_template, request

import data
import graph
import main

app = Flask(__name__)

firstLoad = True

@app.route("/")
def renderLandingPage(errorMessage=None):
    #filling out data on first load of website, otherwise blank
    main.returnBayesEstimates()

    maint = data.cleanMaintenanceData()
    maint = data.reshapeMaintenanceData(maint)

    maint.rename(columns={"date" : "Date", "flight_hours" : "Flight Hours", "system" : "System" , "subsystem" : "Subsystem", "failure_type" : "Failure Type"}, inplace=True)
    maint.sort_values(by="Date", ascending=False, inplace=True)
    maint = maint.iloc[:10]
    maint = maint.to_html(index=False)

    subSystemRatios = main.findWorstPerformingSubSystems()

    return render_template("landing.html", ratios=subSystemRatios, error=errorMessage, recentEntries=maint)

@app.route("/tables")
def serveTables():
    bayesEstimates = main.returnBayesEstimates()
    bayesEstimates.insert(0, "Subsystem", [x for x in range(1,30)])
    bayesEstimates = main.applyEstimatesStyle(bayesEstimates)

    bayesFactors = main.returnBayesFactor()

    return render_template("tables.html", bayesEstimateTable=bayesEstimates, bayesFactorTable=bayesFactors)

@app.route("/graphs")
def serveGraphs():
    graphDir = os.path.join(app.static_folder, "graphs")
    files = sorted(os.listdir(graphDir), key=lambda x: float(x.split('_')[0]))
    fileNames = []
    for f in files:
        name, _ = os.path.splitext(f)
        subsystem, failureType = name.split('_', 1)
        fileNames.append({
            "fname" : f,
            "subsystem" : int(float(subsystem)),
            "failureType" : failureType
        })

    return render_template("graphs.html", graphs=fileNames)

@app.route("/add", methods=["POST"])
def addData():
    print("Received POST request")
    date = request.form["Date"]

    hours = request.form["FlightHours"]
    if not hours or int(hours) < 0 or int(hours) > 24:
        return renderLandingPage(errorMessage="Error, hours cannot be negative or greater than 24, Data Not Added")

    system = request.form["System"]

    subSystem = request.form["SubSystem"]
    if not subSystem or int(subSystem) < 1 or int(subSystem) > 29:
        return renderLandingPage(errorMessage="Error, only subsystems 1-29 are supported, Data Not Added")

    failureType = request.form["FailureType"]
    if not failureType or int(failureType) not in (1,2,6):
        return renderLandingPage(errorMessage="Error, Failure Type can only be 1,2, or 6, Data Not Added")
    data.addEntryToData(date, hours, system, subSystem, failureType)
    return renderLandingPage()

@app.route("/undo", methods=["POST"])
def undoLastEntry():
    data.undoEntry()
    return renderLandingPage()

@app.route("/data")
def serveData():
    originalMaintenanceData = data.cleanMaintenanceData()
    originalMaintenanceData = data.reshapeMaintenanceData(originalMaintenanceData)

    originalMaintenanceData.rename(columns={"date" : "Date", "flight_hours" : "Flight Hours", "system" : "System" , "subsystem" : "Subsystem", "failure_type" : "Failure Type"}, inplace=True)
    
    originalMaintenanceData.sort_values(by="Date", ascending=False, inplace=True)

    originalMaintenanceData = originalMaintenanceData.to_html(index=False)


    originalContractorEstimates = data.cleanContractorData()
    originalContractorEstimates.rename(columns={"SubSystem" : "Subsystem"}, inplace=True)
    originalContractorEstimates = originalContractorEstimates.to_html(index=False)

    return render_template("originalData.html", maintenance=originalMaintenanceData, contractor=originalContractorEstimates)


