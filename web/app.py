import os
import pandas as pd
from flask import Flask, render_template, request

import data
import graph
import main


app = Flask(__name__)

@app.route("/")
def renderLandingPage():
    subSystemRatios = main.findWorstPerformingSubSystems()
    return render_template("landing.html", ratios=subSystemRatios)

@app.route("/tables")
def serveTables():
    bayesEstimates = main.returnBayesEstimates()
    bayesEstimates = bayesEstimates.to_html()

    bayesFactors = main.returnBayesFactor()
    return render_template("tables.html", bayesEstimateTable=bayesEstimates, bayesFactorTable=bayesFactors)

@app.route("/graphs")
def serveGraphs():
    graphDir = os.path.join(app.static_folder, "graphs")
    files = sorted(os.listdir(graphDir), key=lambda x: float(x.split('_')[0]))
    return render_template("graphs.html", graphs=files)

@app.route("/add", methods=["POST"])
def addData():
    print("Received POST request")
    date = request.form["Date"]
    hours = request.form["FlightHours"]
    system = request.form["System"]
    subSystem = request.form["SubSystem"]
    failureType = request.form["FailureType"]
    data.addEntryToData(date, hours, system, subSystem, failureType)
    return render_template("add.html", added="Entry Added", undone="")

@app.route("/undo", methods=["POST"])
def undoLastEntry():
    data.undoEntry()
    return render_template("add.html", added="", undone="Entry Undone")

@app.route("/data")
def serveData():
    originalMaintenanceData = data.cleanMaintenanceData()
    originalMaintenanceData = data.reshapeMaintenanceData(originalMaintenanceData)
    originalMaintenanceData = originalMaintenanceData.to_html()

    originalContractorEstimates = data.cleanContractorData()
    originalContractorEstimates = originalContractorEstimates.to_html()

    return render_template("originalData.html", maintenance=originalMaintenanceData, contractor=originalContractorEstimates)


