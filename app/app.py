import mlflow
import pandas as pd
import flask_monitoringdashboard as dashboard

from joblib import load
from jinja2 import Environment
from werkzeug.serving import run_simple
from jinja2.loaders import FileSystemLoader
from prometheus_client import make_wsgi_app
from flask_prometheus_metrics import register_metrics
from flask import Flask, render_template, redirect, Response
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from app.pipeline_api import data_distr, corr_matrix, model_stats, plot_roc, prec_rec, lime_expl, anchor_expl, shap_expl, tree_expl, retrain_model

app = Flask(__name__)
dashboard.config.init_from(file='app/config.cfg')
dashboard.bind(app)

classifier = load("app/models/classifier.pkl")
X_train = pd.read_csv('app/data/X_train.csv')
X_test = pd.read_csv('app/data/X_test.csv')
Y_train = pd.read_csv('app/data/Y_train.csv')
Y_test = pd.read_csv('app/data/Y_test.csv')

mlflow.set_tracking_uri('app/mlruns')

@app.route("/", methods=['GET', 'POST'])
def disp_home():
    
    print("AHAD")
    return render_template('index.html')

@app.route("/data_distribution")
def data_distribution():

    data_distr(data=X_train.iloc[:,1:], figsizes=(15, 40), cols=6)
    corr_matrix(X_train.iloc[:,1:])

    return render_template('data_dist.html')

@app.route("/model_behaviour")
def model_behaviour():

    cm, acc, prcs, rcl, f1, corcoeff = model_stats(classifier, X_test.iloc[:,1:], Y_test.iloc[:,1])
    anch, prd, prc, cvr = anchor_expl(X_train.iloc[:,1:], X_test.iloc[:,1:], classifier)
    prec_rec(classifier, X_test.iloc[:,1:], Y_test.iloc[:,1])
    plot_roc(classifier, X_test.iloc[:,1:], Y_test.iloc[:,1:])
    lime_expl(X_train.iloc[:,1:], X_test.iloc[:,1:], Y_test.iloc[:,1:], classifier)
    tree_expl(classifier, X_train.iloc[:,1:])
    shap_expl(X_train.iloc[:,1:], X_test.iloc[:,1:], classifier)


    return render_template('model_stat.html', cm=cm, acc=round(acc*100,2), prcs=prcs, rcl=rcl, f1=f1, mcc=corcoeff, anch=anch, prd=prd, prc=prc, cvr=cvr)

@app.route("/retrain")
def retrain():
    
    retrain_model()
    return render_template('retrain.html')

@app.route("/model_monitoring")
def model_monitoring():
    
    return redirect('http://localhost:5000')

register_metrics(app, app_version="v0.1.2", app_config="staging")
dispatcher = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})
run_simple(hostname="localhost", port=5000, application=dispatcher)