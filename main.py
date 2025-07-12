import funciones as func

def main():

    df = func.load_data()
    x_train, x_test, y_train, y_test = func.split_wine_data(df)
    func.mlflow_tracking_xgboost('XGBoost',
                                 x_train, x_test, y_train, y_test,
                                 n_estimators=[100, 300])
    
if __name__ == '__main__':
    main()
