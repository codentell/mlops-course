import requests

data = {"data": [[0.0381, 0.0507, 0.0617, 0.0219, -0.0442, -0.0348, -0.0434, -0.0026, 0.0199, -0.0176]]}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=data)
print(response.json())