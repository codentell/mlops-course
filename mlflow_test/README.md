docker build -t mlflow-app .

docker run -dp 5001:5000 mlflow-app

minikube service mlflow-service

kubectl scale deployment mlflow-deployment --replicas=3