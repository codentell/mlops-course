FROM python:3.11-slim

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]