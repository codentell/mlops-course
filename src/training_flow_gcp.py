from metaflow import FlowSpec, step, Parameter, conda_base, project, retry, timeout, catch, kubernetes

@project(name="lab7_gcp")
@conda_base(
    libraries={
        'numpy': '1.23.5',
        'pandas': '1.4.4',
        'scikit-learn': '1.2.2',
        'mlflow': '2.5.0',
        "databricks-cli": "0.17.6",  # or latest stable version
        "google-cloud-secret-manager": "2.7.0"
    },
    python='3.10.9'
)
class Trainflow(FlowSpec):

    seed = Parameter("seed", default=42)

    @catch(var="start_error")
    @retry(times=2)
    @step
    def start(self):
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=self.seed
        )
        self.next(self.train_random_forest, self.train_gradient_boosting)

    @catch(var="rf_error")
    @retry(times=2)
    @timeout(seconds=300)
    @kubernetes(cpu=1, memory=4000)
    @step
    def train_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error

        model = RandomForestRegressor(random_state=self.seed)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)

        self.model_name = "RandomForest"
        self.mse = mse
        self.model = model
        self.next(self.choose_model)

    @catch(var="gb_error")
    @retry(times=2)
    @timeout(seconds=300)
    @kubernetes(cpu=1, memory=4000)
    @step
    def train_gradient_boosting(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error

        model = GradientBoostingRegressor(random_state=self.seed)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)

        self.model_name = "GradientBoosting"
        self.mse = mse
        self.model = model
        self.next(self.choose_model)

    @retry(times=2)
    @kubernetes(cpu=1, memory=4000)
    @step
    def choose_model(self, inputs):
        from sklearn.metrics import mean_squared_error
        import mlflow

        mlflow.set_tracking_uri("https://mlflow-tracking-174134742093.us-west2.run.app")
        mlflow.set_experiment("lab-7-experiment-gcp")

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

        with mlflow.start_run(run_name=f"register_{best_name}_gcp"):
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="metaflow_train",
                registered_model_name="metaflow-diabetes-gcp-model",
            )

        self.next(self.end)

    @retry(times=2)
    @kubernetes(cpu=1, memory=4000)
    @step
    def end(self):
        print("✅ Model MSEs:")
        for name, mse in self.results:
            print(f"{name}: MSE = {mse:.4f}")
        print(f"\n🏆 Best Model: {self.best_model_name} with MSE = {self.best_mse:.4f}")
        print("🎉 Training and registration complete.")

if __name__ == "__main__":
    Trainflow()
