from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
from sklearn.datasets import load_diabetes
from dataprocessing import load_and_introduce_missing, impute_data
from sklearn.metrics import mean_squared_error




class Trainflow(FlowSpec):

    seed = Parameter('seed', default=42)

    @step
    def start(self):
        """
        Start the preprocessing
        """
        from sklearn.model_selection import train_test_split
        data = load_diabetes()
        df = load_and_introduce_missing(data)
        X, y = impute_data(df)
        mask = ~pd.isna(y)
        X = X[mask]
        y = y[mask]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=self.seed
        )

        self.next(self.train_random_forest, self.train_gradient_boosting)


    @step
    def train_random_forest(self):
        """
        Train Random Forest Regressor
        """
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=self.seed)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)

        self.model_name = "RandomForest"
        self.mse = mse
        self.model = model
        self.next(self.choose_model)

    @step
    def train_gradient_boosting(self):
        """
        """
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(random_state=self.seed)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)

        self.model_name = "GradientBoosting"
        self.mse = mse
        self.model = model
        self.next(self.choose_model)


    @step 
    def choose_model(self, inputs):
        mlflow.set_tracking_uri('https://mlflow-tracking-174134742093.us-west2.run.app')
        mlflow.set_experiment('lab-6-experiment')

        def score(inp):
            preds = inp.model.predict(inp.X_test)
            mse = mean_squared_error(inp.y_test, preds)
            return inp.model, mse, inp.model_name

        sorted_results = sorted(map(score, inputs), key=lambda x: x[1])
        best_model, best_mse, best_name = sorted_results[0]

        self.model = best_model
        self.best_mse = best_mse
        self.best_model_name = best_name
        self.results = [(name, mse) for _, mse, name in sorted_results]

        with mlflow.start_run(run_name=f"register_{best_name}"):
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="metaflow_train",
                registered_model_name="metaflow-diabetes-model"
            )
        mlflow.end_run()

        self.next(self.end)

    @step
    def end(self):
       print('✅ Model MSEs:')
       for name, mse in self.results:
            print(f"{name}: MSE = {mse:.4f}")
       print(f"\n🏆 Best Model: {self.best_model_name} with MSE = {self.best_mse:.4f}")
       print("🎉 Training and registration complete.")

if __name__ == '__main__':
    Trainflow()