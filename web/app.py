import os
import pandas as pd
from flask import Flask, render_template, request

import data
import graph
import main


app = Flask(__name__)

@app.route("/")
def renderLandingPage():
    table3 = main.returnBayesEstimates()
    subSystemRatios = main.findWorstPerformingSubSystems()
    return render_template("landing.html", ratios=subSystemRatios)

@app.route("/table3")
def serveTable3():
    res = main.returnBayesEstimates()
    res = res.to_html()
    return render_template("table3.html", table=res)

@app.route("/table6")
def serverTable6():
    res = main.returnBayesFactor()
    return render_template("table6.html", table=res)


@app.route("/graphs")
def serveGraphs():
    graphDir = os.path.join(app.static_folder, "graphs")
    files = sorted(os.listdir(graphDir), key=lambda x: float(x.split('_')[0]))
    return render_template("graphs.html", graphs=files)

@app.route("/add", methods=["GET", "POST"])
def addData():
    print(request.method)
    if request.method == "GET":
        return render_template("add.html")

    elif request.method == "POST":
        print("Received POST request")
        date = request.form["Date"]
        hours = request.form["FlightHours"]
        system = request.form["System"]
        subSystem = request.form["SubSystem"]
        failureType = request.form["FailureType"]
        data.addEntryToData(date, hours, system, subSystem, failureType)
        return "Entry added"
    else:
        return "Visit / and submit the form"

@app.route("/undo", methods=["POST"])
def undoLastEntry():
    data.undoEntry()
    return "Undid entry"

