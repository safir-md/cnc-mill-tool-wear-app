# cnc-mill-tool-wear-app

## Project Details:
This is a voluntary project. I have created a Flask application in which, I am performing following things:
1. Process Dataset: CNC Mill Tool Wear (https://www.kaggle.com/shasun/tool-wear-detection-in-cnc-mill)
2. Exploratory Data Analysis on the Dataset
3. Create several ML models to train the processed Dataset
4. Develop monitoring Dashboards (usnig Grafana)
5. Monitor the Data and ML models
6. Monitor the Application
7. Expose a few Endpoints
8. Docker Containerize the application
9. Deploy the application on Heroku

## Project Team Members: 
Saﬁr Mohammad Shaikh

## Technologies Used:
* Language: Python
* ML Models: 
1. Decision Tree
2. Random Forest
* Monitoring: 
1. Flask Monitoring Dashboard (https://github.com/flask-dashboard/Flask-MonitoringDashboard)
2. Prometheus, Grafana
* Containerization: Docker, Docker Compose

## How to Run?
1. Clone the Repository:
```sh
git clone https://github.com/Safir-Mohammad-Mustak-Shaikh/cnc-mill-tool-wear-app.git
```
2. Push this repository to your GitHub account.

3. Login to Heroku. You should see the Dashboard.
4. Select New -> create new app -> Enter app name -> Select Region -> Next
5. In Deployment Method, select GitHub -> Authorize access and select this uploaded repo to your Git.
6. Now, the app is deployed to Heroku. Go to Personal -> app -> Select "Open app"
7. Your app is online⭐.
8. To take down the app, run following command through Heroku CLI:
```sh
heroku ps:scale web=0 --app cnc-mill-tool-wear-app
heroku ps:stop "cnc-mill-tool-wear-app" --app cnc-mill-tool-wear-app
```
***Note:** This project is still under development, however, things mentioned above will work fine! Happy Analyzing :-)
