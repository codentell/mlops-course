```bash
kubectl edit deployment argo-server -n argo
```

```yaml
      containers:
      - args:
        - server
        - --auth-mode
        - server
        image: quay.io/argoproj/argocli:v3.5.4
        imagePullPolicy: Always
        name: argo-server
        ports:
        - containerPort: 2746
          name: web
          protocol: TCP
        readinessProbe:
          failureThreshold: 3
          httpGet:
            path: /
            port: 2746
            scheme: HTTPS
          initialDelaySeconds: 10
          periodSeconds: 20
          successThreshold: 1
          timeoutSeconds: 1
        resources: {}
```

```bash
kubectl rollout restart deployment argo-server -n argo
```

```bash
python ./Desktop/mlops-course/metaflow-tools/scripts/forward_metaflow_ports.py --include-argo



python ./Desktop/mlops-course/src/training_flow_gcp.py --environment=conda run

python ./Desktop/mlops-course/src/training_flow_gcp.py  --environment=conda run --with kubernetes

python ./Desktop/mlops-course/src/scoring_flow_gcp.py  --environment=conda run --with kubernetes



```