import mlflow

def mlflow_it(pipeline, X_train, y_train, X_test, y_test):
    estimator_name = pipeline.steps[-1][0]
    scaler_name = pipeline.named_steps['columntransformer'].transformers_[-1][0]
    run_name = f"{estimator_name}_{scaler_name}"
    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, run_name)
        model_uri = mlflow.get_artifact_uri(run_name)
        eval_data = X_test.copy()
        eval_data["label"] = y_test
        result = mlflow.evaluate(
                    model_uri,
                    eval_data,
                    targets="label",
                    model_type="regressor",
                    evaluators=["default"],
                )
        mlflow.end_run()

