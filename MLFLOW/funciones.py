import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_wine
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score


## función carga dataset
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    df['target'] = wine['target']
    return df


## función matriz de correlación
def corr_matrix(df):

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Matriz de correlación del dataset Wine")
    plt.show()

## función de histogramas
def histogram(df):

    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue='target', kde=True, element='step')
        plt.title(f'Distribución de {col} por clase')
        plt.show()

## función de diagrama de pares
def d_pares(df):
    sns.pairplot(df, hue='target', vars=df.columns[:4])
    plt.show()


## función separa datos
def split_wine_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test



def mlflow_tracking_xgboost(nombre_job, x_train, x_test, y_train, y_test, n_estimators):
    mlflow.set_experiment(nombre_job)

    for n in n_estimators:
        with mlflow.start_run(run_name=f"xgb_{n}"):
            xgb = XGBRegressor(
                n_estimators=n,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                n_jobs=-1
            )

            xgb.fit(x_train, y_train)

            y_pred_train = xgb.predict(x_train)
            y_pred_test = xgb.predict(x_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)

            mlflow.log_param('n_estimators', n)
            mlflow.log_metrics({
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'r2_test': r2_test
            })

            mlflow.sklearn.log_model(xgb, artifact_path='xgb_model')

    print("Entrenamiento XGBoost terminado correctamente")


