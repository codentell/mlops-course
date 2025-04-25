
```bash
gcloud init mlops-lab-9

gcloud config set project mlops-lab-9




```bash
docker build -t reddit-app:latest .
docker run -it -p 8000:8000 reddit-app:latest
```



```bash
minikube start
minikube dashboard
```

```bash
eval $(minikube docker-env)
docker build -t reddit-app:latest .
docker images  # Confirm it's built inside Minikube

```

```bash
kubectl apply -f reddit-deployment.yaml
kubectl apply -f reddit-service.yaml
```

```bash
minikube service reddit-service --url
```

GKE
```bash
gcloud services enable container.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

```bash
gcloud artifacts repositories create lab9 \
  --repository-format=docker \
  --location=us-west1
```

```bash
gcloud auth configure-docker us-west1-docker.pkg.dev

docker tag reddit-app:latest us-west1-docker.pkg.dev/mlops-lab-9/lab9/reddit-app:latest

docker push us-west1-docker.pkg.dev/mlops-lab-9/lab9/reddit-app:latest

```

```bash
gcloud container clusters create reddit-cluster --num-nodes=3 --region=us-west2 --disk-size=50
```

```bash
gcloud container clusters create-auto reddit-cluster --region=us-west1
```


```bash
gcloud container clusters get-credentials reddit-cluster --region=us-west1
```

```bash
docker buildx create --use   # Only needed once to enable Buildx

docker buildx build \
  --platform linux/amd64 \
  -t reddit-app:latest \
  --load .

```

```bash
docker tag reddit-app:latest us-west1-docker.pkg.dev/mlops-lab-9/lab9/reddit-app:latest
docker push us-west1-docker.pkg.dev/mlops-lab-9/lab9/reddit-app:latest
```

```bash
kubectl rollout restart deployment reddit-app
```

```bash
kubectl get pods
kubectl logs <new-pod-name>
```


```bash
kubectl apply -f reddit-deployment.yaml
kubectl apply -f reddit-service.yaml
```

```bash
kubectl get service reddit-service
```


