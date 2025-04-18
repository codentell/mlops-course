from metaflow import FlowSpec, step, Parameter, kubernetes, retry, catch, timeout, conda_base

@conda_base(python='3.9.16', libraries={
    "pandas": "1.4.4",
    "scikit-learn": "1.2.2",
    "mlflow": "2.9.2",
    "databricks-cli": "0.17.6"  # to match Trainflow, prevents registry import issues
})
class ScoringFlow(FlowSpec):

    seed = Parameter("seed", default=123)

    @catch(var="start_error")
    @retry(times=2)
    @timeout(seconds=180)
    @kubernetes(cpu=1, memory=4000)
    @step
    def start(self):
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        data = load_diabetes()
        X, y = data.data, data.target

        _, self.X_test, _, self.y_test = train_test_split(X, y, random_state=self.seed)

        print(f"Holdout test data: {self.X_test.shape[0]} samples")
        self.next(self.load_model)

    @catch(var="load_model_error")
    @retry(times=2)
    @timeout(seconds=180)
    @kubernetes(cpu=1, memory=4000)
    @step
    def load_model(self):
        import mlflow
        mlflow.set_tracking_uri("https://mlflow-tracking-174134742093.us-west2.run.app")

        model_uri = 'runs:/9becda7e7f2c458989dd6235cf570924/metaflow_train'
        self.model = mlflow.pyfunc.load_model(model_uri)

        print("✅ Model loaded from MLflow Registry (Production stage)")
        self.next(self.predict)

    @catch(var="predict_error")
    @retry(times=2)
    @timeout(seconds=180)
    @kubernetes(cpu=1, memory=4000)
    @step
    def predict(self):
        from sklearn.metrics import mean_squared_error
        preds = self.model.predict(self.X_test)
        self.preds = preds
        self.mse = mean_squared_error(self.y_test, preds)
        self.next(self.end)

    @retry(times=2)
    @kubernetes(cpu=1, memory=4000)
    @step
    def end(self):
        import pandas as pd
        print(f"🎯 Final Scoring MSE: {self.mse:.4f}")
        print("📊 Sample Predictions:")
        print(pd.DataFrame({
            "y_true": self.y_test[:5],
            "y_pred": self.preds[:5]
        }))

if __name__ == "__main__":
    ScoringFlow()
