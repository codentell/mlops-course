from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
from sklearn.datasets import load_diabetes
from dataprocessing import load_and_introduce_missing, impute_data
from sklearn.metrics import mean_squared_error


class ScoringFlow(FlowSpec):

    seed = Parameter("seed", default=123)  # Use a different seed from training

    @step
    def start(self):
        """
        Load new holdout data using the same preprocessing pipeline.
        """
        from sklearn.model_selection import train_test_split

        # Load & transform
        data = load_diabetes()
        df = load_and_introduce_missing(data)
        X, y = impute_data(df)

        # Remove missing targets
        mask = ~pd.isna(y)
        X, y = X[mask], y[mask]

        # Generate a new test set
        _, self.X_test, _, self.y_test = train_test_split(X, y, random_state=self.seed)

        print(f"Holdout test data: {self.X_test.shape[0]} samples")
        self.next(self.load_model)

    @step
    def load_model(self):
        """
        Load the best model from the MLflow Model Registry.
        """
        mlflow.set_tracking_uri("https://mlflow-tracking-174134742093.us-west2.run.app")

        model_uri = 'runs:/b062a37fda4a4f61bc6e77a9cf059345/metaflow_train'
        self.model = mlflow.pyfunc.load_model( model_uri )

        print("Model loaded from MLflow Registry (Production stage)")
        self.next(self.predict)

    @step
    def predict(self):
        """
        Run predictions and compute MSE.
        """
        preds = self.model.predict(self.X_test)
        self.preds = preds
        self.mse = mean_squared_error(self.y_test, preds)

        self.next(self.end)

    @step
    def end(self):
        print(f"Final Scoring MSE: {self.mse:.4f}")
        print("Sample Predictions:")
        print(pd.DataFrame({
            "y_true": self.y_test[:5],
            "y_pred": self.preds[:5]
        }))


if __name__ == "__main__":
    ScoringFlow()
