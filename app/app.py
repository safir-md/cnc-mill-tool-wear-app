import pandas as pd

from flask import Flask, render_template
from joblib import load

from pipeline_api import data_distr, corr_matrix, model_stats, plot_roc, prec_rec

app = Flask(__name__)
rf_classifier = load("./models/rf_model.pkl")
dt_classifier = load("./models/dt_model.pkl")
X_train_DT = pd.read_csv('./data/X_train_DT.csv')
X_test_DT = pd.read_csv('./data/X_test_DT.csv')
X_train_RF = pd.read_csv('./data/X_train_RF.csv')
X_test_RF = pd.read_csv('./data/X_test_RF.csv')
Y_train = pd.read_csv('./data/Y_train.csv')
Y_test = pd.read_csv('./data/Y_test.csv')

@app.route("/")
def disp_home():
    
    print("AHAD")
    return "Allahu Akbar"

@app.route("/data_distribution")
def data_distribution():
    
    #data_distr(data=X_train_DT, figsizes=(15, 40), cols=6)
    data_distr(data=X_train_RF.iloc[:,1:], figsizes=(15, 40), cols=6)

    #corr_matrix(X_train_DT)
    corr_matrix(X_train_RF.iloc[:,1:])

    return render_template('data_dist.html')

@app.route("/model_behaviour")
def model_behaviour():

    cm_dt, acc_dt, f1_dt, corcoeff_dt = model_stats(dt_classifier, X_test_DT.iloc[:,1:], Y_test.iloc[:,1])
    cm, acc, f1, corcoeff = model_stats(rf_classifier, X_test_RF.iloc[:,1:], Y_test.iloc[:,1])

    plot_roc(dt_classifier, X_test_DT.iloc[:,1:], Y_test.iloc[:,1])
    prec_rec(rf_classifier, X_test_RF.iloc[:,1:], Y_test.iloc[:,1])

    return render_template('model_stat.html', cm=cm, acc=acc, f1=f1, mcc=corcoeff)
