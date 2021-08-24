import pandas as pd
import flask_monitoringdashboard as dashboard

from flask import Flask, render_template
from joblib import load

from app.pipeline_api import data_distr, corr_matrix, model_stats, plot_roc, prec_rec

app = Flask(__name__)
dashboard.config.init_from(file='app/config.cfg')
dashboard.bind(app)
classifier = load("app/models/classifier.pkl")
X_train = pd.read_csv('app/data/X_train.csv')
X_test = pd.read_csv('app/data/X_test.csv')
Y_train = pd.read_csv('app/data/Y_train.csv')
Y_test = pd.read_csv('app/data/Y_test.csv')

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

    cm, acc, f1, corcoeff = model_stats(classifier, X_test.iloc[:,1:], Y_test.iloc[:,1])
    prec_rec(classifier, X_test.iloc[:,1:], Y_test.iloc[:,1])

    return render_template('model_stat.html', cm=cm, acc=round(acc*100,2), f1=f1, mcc=corcoeff)
