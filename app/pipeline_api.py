import os
import lime
import shap
import mlflow
import mlflow.sklearn
import lime.lime_tabular

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from anchor import utils
from anchor import anchor_tabular
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import matthews_corrcoef, f1_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, plot_precision_recall_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV

mlflow.set_tracking_uri('app/mlruns')

def fetch_data():
    dataset = pd.read_csv('app/data/train.csv')
    frames = []
    for ctr in range(1,19):
        num = '0' + str(ctr) if ctr < 10 else str(ctr)
        frame = pd.read_csv("app/data/experiment_{}.csv".format(num))
        row = dataset[dataset['No'] == ctr]
        frame['clamp_pressure'] = row.iloc[0]['clamp_pressure']
        frame['tool_condition'] = row.iloc[0]['tool_condition']
        frames.append(frame)
    dataframe = pd.concat(frames, ignore_index = True)
    print("Data Fetched")
    return dataframe

def feat_engg(dataframe):
    dataframe = dataframe[(dataframe.Machining_Process == 'Layer 1 Up') | 
                        (dataframe.Machining_Process == 'Layer 1 Down') |
                        (dataframe.Machining_Process == 'Layer 2 Up') | 
                        (dataframe.Machining_Process == 'Layer 2 Down') |
                        (dataframe.Machining_Process == 'Layer 3 Up') | 
                        (dataframe.Machining_Process == 'Layer 3 Down')]

    dataframe['X1_DiffPosition'] = dataframe['X1_CommandPosition'] - dataframe['X1_ActualPosition']
    dataframe['X1_DiffVelocity'] = dataframe['X1_CommandVelocity'] - dataframe['X1_ActualVelocity']
    dataframe['X1_DiffAcceleration'] = dataframe['X1_CommandAcceleration'] - dataframe['X1_ActualAcceleration']
    dataframe['Y1_DiffPosition'] = dataframe['Y1_CommandPosition'] - dataframe['Y1_ActualPosition']
    dataframe['Y1_DiffVelocity'] = dataframe['Y1_CommandVelocity'] - dataframe['Y1_ActualVelocity']
    dataframe['Y1_DiffAcceleration'] = dataframe['Y1_CommandAcceleration'] - dataframe['Y1_ActualAcceleration']
    dataframe['Z1_DiffPosition'] = dataframe['Z1_CommandPosition'] - dataframe['Z1_ActualPosition']
    dataframe['Z1_DiffVelocity'] = dataframe['Z1_CommandVelocity'] - dataframe['Z1_ActualVelocity']
    dataframe['Z1_DiffAcceleration'] = dataframe['Z1_CommandAcceleration'] - dataframe['Z1_ActualAcceleration']
    dataframe['S1_DiffPosition'] = dataframe['S1_CommandPosition'] - dataframe['S1_ActualPosition']
    dataframe['S1_DiffVelocity'] = dataframe['S1_CommandVelocity'] - dataframe['S1_ActualVelocity']
    dataframe['S1_DiffAcceleration'] = dataframe['S1_CommandAcceleration'] - dataframe['S1_ActualAcceleration']

    drop_cols = ['X1_CommandPosition', 'X1_CommandVelocity',
                    'X1_CommandAcceleration', 'Y1_CommandPosition',
                    'Y1_CommandVelocity', 'Y1_CommandAcceleration',
                    'Z1_CommandPosition', 'Z1_CommandVelocity',
                    'Z1_CommandAcceleration', 'S1_CommandPosition',
                    'S1_CommandVelocity', 'S1_CommandAcceleration',
                    'Machining_Process']
    dataframe = dataframe.drop(drop_cols, axis=1)

    dummies_tc = pd.get_dummies(dataframe.tool_condition)
    dataframe = pd.concat([dataframe, dummies_tc], axis='columns')
    dataframe = dataframe.drop(['unworn', 'tool_condition'], axis=1)

    drop_cols = ['S1_SystemInertia', 'Z1_OutputVoltage', 'Z1_OutputCurrent',
                    'Z1_DCBusVoltage', 'Z1_CurrentFeedback']
    dataframe = dataframe.drop(drop_cols, axis=1)
    print("Feature Engineering Done")
    return dataframe

def split_data(dataframe):
    y = dataframe['worn']
    X = dataframe.drop(['worn'], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    Y_train.to_csv('app/data/Y_train.csv')
    Y_test.to_csv('app/data/Y_test.csv')
    print("Data Splitted")
    return X, X_train, X_test, Y_train, Y_test

def feat_select(classifier, X, X_train, X_test):
    importances = list(classifier.feature_importances_)
    feature_list = list(X.columns)
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances) if round(importance, 4)>=0.02]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    importances.sort(reverse=True)
    cumulative_importances = np.cumsum(importances)
    feature_count = np.where(cumulative_importances > 0.95)[0][0] + 1
    # Number of Features to keep
    feature_count = feature_count if feature_count<=len(feature_importances) else len(feature_importances)
    imp_feature_names = [feature[0] for feature in feature_importances[0:feature_count]]

    X_train_NEW = X_train.loc[:, imp_feature_names]
    X_test_NEW = X_test.loc[:, imp_feature_names]
    print("Features Selected")
    return X_train_NEW, X_test_NEW

def dec_tr_model(X, X_train, X_test, Y_train):
    classifier_dt_1 = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
    classifier_dt_1.fit(X_train, Y_train)

    X_train_DT, X_test_DT = feat_select(classifier_dt_1, X, X_train, X_test)
    X_train_DT.to_csv('app/data/X_train_DT.csv')
    X_test_DT.to_csv('app/data/X_test_DT.csv')

    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]                                  # Max Levels in the Tree
    max_depth.append(None)
    max_features = ['auto', 'sqrt', 'log2', None]                                                 # Number of Features at every split
    min_samples_split = [2, 4, 6, 8, 10]
    min_samples_leaf = [1, 2, 4, 8, 12]
    criterion = ['gini', 'entropy']                                                               # Function to measure the quality of a split

    random_grid = {'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'criterion': criterion}

    base_dt_model = DecisionTreeClassifier()

    mlflow.set_tracking_uri('app/mlruns')
    dec_exp_id = mlflow.set_experiment("Decision-Tree Experiment")
    with mlflow.start_run(experiment_id=dec_exp_id):
        random_dt_model = RandomizedSearchCV(estimator=base_dt_model,
                                                param_distributions=random_grid,
                                                n_iter=200, cv=5, verbose=2,
                                                random_state=0, n_jobs=-1)
        classifier_dt_2 = random_dt_model.fit(X_train_DT, Y_train)

        mlflow.log_params(random_grid)
        mlflow.sklearn.log_model(random_dt_model.best_estimator_, "DT Classifier")

    pkl_file = "app/models/dt_model.pkl"
    dump(random_dt_model.best_estimator_, pkl_file)
    print("Model Trained")

    return X_test_DT, random_dt_model.best_estimator_

def rf_model(X, X_train, X_test, Y_train):
    classifier_rf_1 = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state = 0)
    classifier_rf_1.fit(X_train, Y_train)

    X_train_RF, X_test_RF = feat_select(classifier_rf_1, X, X_train, X_test)
    X_train_RF.to_csv('app/data/X_train_RF.csv')
    X_test_RF.to_csv('app/data/X_test_RF.csv')

    n_estimators = [int(x) for x in np.linspace(200, 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    max_features = ['auto', 'sqrt', None]
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [1, 2, 4, 8]
    criterion = ['gini', 'entropy']
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'criterion': criterion,
                    'bootstrap': bootstrap}

    base_rf_model = RandomForestClassifier()

    mlflow.set_tracking_uri('app/mlruns')
    raf_exp_id = mlflow.set_experiment("Random-Forest Experiment")
    with mlflow.start_run(experiment_id=raf_exp_id):
        random_rf_model = RandomizedSearchCV(estimator=base_rf_model,
                                                param_distributions=random_grid,
                                                n_iter=100, cv=3, verbose=2,
                                                random_state=0, n_jobs=-1)
        classifier_rf_2 = random_rf_model.fit(X_train_RF, Y_train)

        mlflow.log_params(random_grid)
        mlflow.sklearn.log_model(random_rf_model.best_estimator_, "RF Classifier")

    pkl_file = "app/models/rf_model.pkl"
    dump(random_rf_model.best_estimator_, pkl_file)
    print("Model Trained")

    return X_test_RF, random_rf_model.best_estimator_

def compare_models(acc_dt, acc_rf):
    if acc_rf>=acc_dt:
        os.remove('app/models/dt_model.pkl')
        os.rename('app/models/rf_model.pkl','app/models/classifier.pkl')
        os.remove('app/data/X_train_DT.csv')
        os.remove('app/data/X_test_DT.csv')
        os.rename('app/data/X_train_RF.csv', 'app/data/X_train.csv')
        os.rename('app/data/X_test_RF.csv', 'app/data/X_test.csv')
    else:
        os.remove('app/models/rf_model.pkl')
        os.rename('app/models/dt_model.pkl','app/models/classifier.pkl')
        os.remove('app/data/X_train_RF.csv')
        os.remove('app/data/X_test_RF.csv')
        os.rename('app/data/X_train_DT.csv', 'app/data/X_train.csv')
        os.rename('app/data/X_test_DT.csv', 'app/data/X_test.csv')

def model_stats(classifier, X_test, Y_test):
    mlflow.set_tracking_uri('app/mlruns')
    fin_exp_id = mlflow.set_experiment("Final Model Experiment")
    with mlflow.start_run(experiment_id=fin_exp_id):
        Y_pred = classifier.predict(X_test)
        cm = confusion_matrix(Y_test, Y_pred)
        acc = round(accuracy_score(Y_test, Y_pred), ndigits=4)
        prc = round(precision_score(Y_test, Y_pred), ndigits=4)
        rcl = round(recall_score(Y_test, Y_pred), ndigits=4)
        f1 = round(f1_score(Y_test, Y_pred), ndigits=4)
        corcoeff = round(matthews_corrcoef(Y_test, Y_pred), ndigits=4)

        mlflow.log_metrics({'Accuracy Score': acc, 'Precision Score': prc, 'Recall Score':rcl, 'F1 Score':f1, 'Correlation Coefficient': corcoeff})
        mlflow.sklearn.log_model(classifier, "Final Classifier")

    print("All Done",acc)
    return cm, acc, prc, rcl, f1, corcoeff

def retrain_model():    
    dataframe = fetch_data()
    dataframe = feat_engg(dataframe)
    dataframe_copy = dataframe
    X, X_train, X_test, Y_train, Y_test = split_data(dataframe)
    X_test_DT, dt_classifier = dec_tr_model(X, X_train, X_test, Y_train)
    _, acc_dt, _, _, _, _ = model_stats(dt_classifier, X_test_DT, Y_test)

    X, X_train, X_test, Y_train, Y_test = split_data(dataframe_copy)
    X_test_RF, rf_classifier = rf_model(X, X_train, X_test, Y_train)
    _, acc_rf, _, _, _, _ = model_stats(rf_classifier, X_test_RF, Y_test)

    compare_models(acc_dt, acc_rf)

"""
Visualization | Graphs | Charts
"""

def prec_rec(classifier, X_test, Y_test):
    Y_test_score = classifier.predict_proba(X_test)[:, 1]
    average_precision = average_precision_score(Y_test, Y_test_score)
    disp = plot_precision_recall_curve(classifier, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
    plt.savefig('app/static/prec_rec.svg', bbox_inches='tight')

def plot_roc(classifier, X_test, y_test):
    y_test_score = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig('app/static/roc.svg', bbox_inches='tight')

def data_distr(data, figsizes, cols, shareys=True, colors='green'):
    ref = 0 
    rows = round(len(data.columns)/cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsizes, sharey=shareys, squeeze=True)

    for n in range(cols):
        for x in range(rows):
            if ref < data.shape[1]:
                axs[x, n].set_title(data.columns[ref])
                axs[x, n].hist(data[data.columns[ref]], color=colors)
                ref += 1
           
    plt.savefig('app/static/dat_dist.svg', bbox_inches='tight')
    #return plt.show()

def corr_matrix(dataframe):
    corr_dataframe = dataframe.corr()
    dissimilarity = 1 - abs(corr_dataframe)
    Z = linkage(squareform(dissimilarity), 'complete')

    # Clusterize the data
    threshold = 0.4
    labels = fcluster(Z, threshold, criterion='distance')

    # Keep the indices to sort labels
    labels_order = np.argsort(labels)

    # Build a new dataframe with the sorted columns
    for idx, i in enumerate(dataframe.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(dataframe[i])
        else:
            df_to_append = pd.DataFrame(dataframe[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)

    # Plot
    plt.figure(figsize=(20,20))
    correlations = clustered.corr()
    sns.heatmap(round(correlations,2),
                cmap=sns.diverging_palette(20, 220, n=200), 
                square=True, 
                annot=False, 
                linewidths=.5,
                vmin=-1, vmax=1, center= 0,
                cbar_kws={"shrink": .5})
    plt.title("Clusterized Correlation Matrix")
    plt.yticks(rotation=0)
    plt.savefig('app/static/cor_mat.svg', bbox_inches='tight')
    #return plt.show()

def lime_expl(X_train, X_test, Y_test, classifier):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.to_numpy(), 
        feature_names=X_train.columns.tolist()
    )

    i = np.random.randint(0, X_test.shape[0])
    explanation = lime_explainer.explain_instance(
        X_test.iloc[i], 
        classifier.predict_proba,
        num_features=5,
        top_labels=1
    )
    print(f"Real value {Y_test.iloc[i]}")
    explanation.save_to_file('app/static/lime_expl.html')

def anchor_expl(X_train, X_test, classifier):
    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=["0", "1"],
        feature_names=X_train.columns.tolist(),
        train_data=X_train.to_numpy(),
    )
    i = np.random.randint(0, X_test.shape[0])
    sample = X_test.iloc[i].to_numpy().reshape(1, -1)
    prediction = classifier.predict(sample)[0]
    anchor_explanation = anchor_explainer.explain_instance(
        X_train.iloc[i].to_numpy(), 
        classifier.predict,
        threshold=0.95
        )
    exp = (" AND ".join(anchor_explanation.names()))
    return exp, anchor_explainer.class_names[prediction], anchor_explanation.precision(), anchor_explanation.coverage()

def shap_expl(X_train, X_test, classifier):
    shap.initjs()
    shap_explainer = shap.KernelExplainer(classifier.predict_proba, X_train[:100])
    shap_values = shap_explainer.shap_values(X_test.iloc[0,:])
    shap.force_plot(shap_explainer.expected_value[0], shap_values[0], X_test.iloc[0,:], show=False, matplotlib=True)
    plt.savefig('app/static/shap_exp_1.svg', bbox_inches='tight')

    """
    shap_values = shap_explainer.shap_values(X_test.iloc[0:10,:])
    shap.force_plot(shap_explainer.expected_value[0], shap_values[0], X_test.iloc[0:10,:], show=False)
    plt.savefig('app/static/shap_exp_2.png', bbox_inches='tight')
    """

def tree_expl(classifier, X_train):
    train_prediction = classifier.predict(X_train)

    gs_explainer = DecisionTreeClassifier(max_depth=3)
    gs_explainer.fit(X_train, train_prediction)
    plt.figure(figsize=(25, 20))
    plot_tree(gs_explainer, feature_names=X_train.columns, class_names=['unworn', 'worn'], fontsize=12)

    plt.savefig('app/static/tree_expl.svg', bbox_inches='tight')
