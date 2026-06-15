import os
from flask import Flask, render_template

import data
import graph
import main


app = Flask(__name__)

@app.route("/")
def helloWorld():
    return render_template("landing.html")

@app.route("/table3")
def serveTables():
    res = main.main()
    return f"<p>{res[0]}</p><p>{res[1]}</p>"

@app.route("/graphs")
def serveGraphs():
    graphDir = os.path.join(app.static_folder, "graphs")
    files = sorted(os.listdir(graphDir))
    return render_template("graphs.html", graphs=files)
