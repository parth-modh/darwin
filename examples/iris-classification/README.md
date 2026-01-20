# Iris Classification with Spark - End-to-End ML Platform Example

This example demonstrates the complete ML lifecycle on the Darwin platform using distributed Spark training for iris species classification.

## Overview

You will learn how to:
1. Set up the Darwin ML platform with required services
2. Create and manage a compute cluster with Spark support
3. Train a Random Forest model using PySpark ML
4. Track experiments and register models with MLflow
5. Deploy models for inference using ML-Serve
6. Test inference endpoints and clean up resources

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Darwin ML Platform                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Compute    │    │    MLflow    │    │   ML-Serve   │               │
│  │   Cluster    │───▶│   Registry   │───▶│  Deployment  │               │
│  │  (Ray+Spark) │    │              │    │              │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Jupyter Lab  │    │   Model      │    │  Inference   │               │
│  │  Notebook    │    │   Artifacts  │    │   Endpoint   │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker installed and running
- `kubectl` CLI installed
- Python 3.9.7+
- At least 8GB RAM available for the local cluster

---

## Step 1: Initialize Platform Configuration

Run the example initialization script to configure the required services:

```bash
# From the project root directory
cd /path/to/darwin

# Run the example init script
sh examples/iris-classification/init-example.sh
```

This enables:
- **Compute**: `darwin-compute`, `darwin-cluster-manager`
- **MLflow**: `darwin-mlflow`, `darwin-mlflow-app`
- **Serve**: `ml-serve-app`, `artifact-builder`
- **Runtime**: `ray:2.37.0` with Darwin SDK (Spark support)
- **CLI**: `darwin-cli`

Alternatively, run `./init.sh` manually and select:
- Compute: Yes
- MLflow: Yes
- Serve: Yes
- Darwin SDK Runtime: Yes
- Ray runtime `ray:2.37.0`: Yes
- Darwin CLI: Yes

---

## Step 2: Build and Deploy Platform

Build all required images and set up the local Kubernetes cluster:

```bash
# Build images (answer 'y' to prompts, or use -y for auto-yes)
./setup.sh -y

# Deploy the platform
./start.sh
```

Wait for all pods to be ready. You can check status with:

```bash
export KUBECONFIG=./.setup/kindkubeconfig.yaml
kubectl get pods -n darwin
```

---

## Step 3: Configure Darwin CLI

Activate the virtual environment and configure the CLI:

```bash
# Activate virtual environment
source .venv/bin/activate

# Configure CLI environment
darwin config set --env darwin-local

# Verify CLI is working
darwin --help
```

---

## Step 4: Create Compute Cluster

Create a compute cluster with Spark support using the provided configuration:

```bash
darwin compute create --file examples/iris-classification/cluster-config.yaml
```

Expected output:
```
Cluster created successfully!
Cluster ID: <CLUSTER_ID>
Name: iris-spark-example
Status: PENDING
```

Save the `CLUSTER_ID` for later steps:

```bash
export CLUSTER_ID=<your-cluster-id>
```

Wait for the cluster to be running:

```bash
# Check cluster status
darwin compute get --cluster-id $CLUSTER_ID
```

Wait until `Status: RUNNING` appears.

---

## Step 5: Access Jupyter Lab

Once the cluster is running, access Jupyter Lab in your browser:

```
http://localhost/kind-0/{CLUSTER_ID}-jupyter/lab
```

Replace `{CLUSTER_ID}` with your actual cluster ID.

---

## Step 6: Run Training Notebook

In Jupyter Lab:

1. Create a new Python 3 notebook or upload `train_iris_model_spark.ipynb`

2. If creating a new notebook, copy the cells from `train_iris_model_spark.ipynb`:

**Cell 1: Install Dependencies**
```python
# Fix pyOpenSSL/cryptography compatibility issue first
%pip install --upgrade pyOpenSSL cryptography

# Install main dependencies
%pip install pandas numpy scikit-learn mlflow pyspark
```

**Cell 2: Import Libraries**
```python
import os
import json
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime

# Spark imports
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRFClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# MLflow imports
import mlflow
import mlflow.spark
from mlflow import set_tracking_uri, set_experiment
from mlflow.client import MlflowClient

# Scikit-learn imports
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

# Darwin SDK imports
import ray
from darwin import init_spark_with_configs, stop_spark

print("Darwin SDK available - will use distributed Spark on Darwin cluster")
```

**Cell 3: Initialize Spark with Darwin SDK**
```python
# Spark configurations
spark_configs = {
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.shuffle.partitions": "4",
    "spark.default.parallelism": "4",
    "spark.executor.memory": "1g",
    "spark.executor.cores": "1",
    "spark.driver.memory": "1g",
    "spark.executor.instances": "2",
}

ray.init()
spark = init_spark_with_configs(spark_configs=spark_configs)
print(f"Spark version: {spark.version}")
```

**Cell 4: Setup MLflow**
```python
MLFLOW_URI = "http://darwin-mlflow-lib.darwin.svc.cluster.local:8080"
USERNAME = "abc@gmail.com"
PASSWORD = "password"
EXPERIMENT_NAME = "iris_spark_classification"
MODEL_NAME = "IrisSparkRFClassifier"

os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD
set_tracking_uri(MLFLOW_URI)
client = MlflowClient(MLFLOW_URI)
set_experiment(experiment_name=EXPERIMENT_NAME)
print(f"MLflow configured: {MLFLOW_URI}")
```

**Cell 5: Load and Prepare Data**
```python
# Load Iris dataset
iris = load_iris(as_frame=True)
pdf = iris.data.copy()
pdf.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
pdf['label'] = iris.target.astype(float)
target_names = iris.target_names.tolist()

# Convert to Spark DataFrame
df = spark.createDataFrame(pdf)
feature_cols = [c for c in df.columns if c != 'label']

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train samples: {train_df.count()}, Test samples: {test_df.count()}")
```

**Cell 6: Train Model**
```python
# Define hyperparameters
hyperparams = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 1,
    "random_state": 42,
}

# Create and train Random Forest
rf_classifier = SparkRFClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=hyperparams["n_estimators"],
    maxDepth=hyperparams["max_depth"],
    seed=hyperparams["random_state"],
)

with mlflow.start_run(run_name=f"spark_rf_iris_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Train model
    model = rf_classifier.fit(train_df)
    print(f"Model trained with {model.getNumTrees} trees")
    
    # Evaluate
    test_pred = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.setMetricName("accuracy").evaluate(test_pred)
    f1 = evaluator.setMetricName("f1").evaluate(test_pred)
    
    # Log to MLflow
    mlflow.log_params(hyperparams)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1", f1)
    
    # Save model artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "spark_model")
        model.save(model_path)
        mlflow.log_artifacts(model_path, artifact_path="model/spark_model")
    
    run_id = mlflow.active_run().info.run_id
    experiment_id = mlflow.active_run().info.experiment_id
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Run ID: {run_id}")
```

**Cell 7: Register Model**
```python
model_uri = f"runs:/{run_id}/model"

# Create registered model if needed
try:
    client.get_registered_model(MODEL_NAME)
    print(f"Model '{MODEL_NAME}' exists")
except:
    client.create_registered_model(MODEL_NAME)
    print(f"Created model: {MODEL_NAME}")

# Register version
result = client.create_model_version(
    name=MODEL_NAME,
    source=model_uri,
    run_id=run_id
)
print(f"Registered {MODEL_NAME} version {result.version}")
print(f"\nModel URI for deployment: models:/{MODEL_NAME}/{result.version}")
```

**Cell 8: Cleanup Spark**
```python
stop_spark()
print("Spark session stopped")
```

3. Run all cells in sequence

4. Note the **Run ID**, **Experiment ID**, and **Model Version** from the output

---

## Step 7: Verify MLflow Model Registration

Back in your terminal, verify the model was registered:

```bash
# List all registered models
darwin mlflow model list

# Get details of the iris model
darwin mlflow model get --name IrisSparkRFClassifier

# Get specific version details
darwin mlflow model get --name IrisSparkRFClassifier --version 1
```

Expected output:
```
Model: IrisSparkRFClassifier
Latest Version: 1
Description: Iris Spark RF Classifier
```

---

## Step 8: Stop the Compute Cluster

After training is complete, stop the cluster to free resources:

```bash
darwin compute stop --cluster-id $CLUSTER_ID
```

Verify the cluster is stopped:

```bash
darwin compute get --cluster-id $CLUSTER_ID
```

---

## Step 9: Create ML-Serve Application

Create a new serve application for the model:

```bash
darwin serve create \
  --name iris-spark-classifier \
  --type api \
  --space ml-examples \
  --description "Iris Spark Random Forest Classifier"
```

---

## Step 10: Configure Serve Infrastructure

Create the infrastructure configuration for the serve:

```bash
darwin serve config create \
  --serve-name iris-spark-classifier \
  --env darwin-local \
  --file examples/iris-classification/serve-config.yaml
```

Or configure inline:

```bash
darwin serve config create \
  --serve-name iris-spark-classifier \
  --env darwin-local \
  --backend-type fastapi \
  --cores 2 \
  --memory 4 \
  --node-capacity ondemand \
  --min-replicas 1 \
  --max-replicas 3
```

---

## Step 11: Build Serve Artifact

Build the deployment artifact:

```bash
darwin serve artifact create \
  --serve-name iris-spark-classifier \
  --version v1.0.0 \
  --github-repo-url https://github.com/your-org/iris-serve-repo \
  --branch main
```

Check build status:

```bash
darwin serve artifact jobs
darwin serve artifact status --job-id <JOB_ID>
```

---

## Step 12: Deploy the Model

Deploy the model using the MLflow model URI:

```bash
darwin serve deploy-model \
  --serve-name iris-spark-classifier \
  --artifact-version v1.0.0 \
  --model-uri models:/IrisSparkRFClassifier/1 \
  --env darwin-local \
  --cores 2 \
  --memory 4 \
  --node-capacity ondemand \
  --min-replicas 1 \
  --max-replicas 3
```

Check deployment status:

```bash
darwin serve status --name iris-spark-classifier --env darwin-local
```

Wait until the status shows `RUNNING`.

---

## Step 13: Test Inference

Test the deployed model with sample requests:

**Using curl:**

```bash
curl -X POST http://localhost/serve/iris-spark-classifier/predict \
  -H "Content-Type: application/json" \
  -d @examples/iris-classification/sample-request.json
```

**Sample request payload:**

```json
{
  "instances": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
  ]
}
```

**Expected response:**

```json
{
  "predictions": [
    {
      "class": 0,
      "species": "setosa",
      "probability": [0.95, 0.03, 0.02]
    }
  ]
}
```

**Test with different iris samples:**

```bash
# Versicolor sample
curl -X POST http://localhost/serve/iris-spark-classifier/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}]}'

# Virginica sample
curl -X POST http://localhost/serve/iris-spark-classifier/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}]}'
```

---

## Step 14: Undeploy the Serve Application

When done, undeploy the serve application:

```bash
darwin serve undeploy-model --serve-name iris-spark-classifier --env darwin-local
```

Verify undeployment:

```bash
darwin serve status --name iris-spark-classifier --env darwin-local
```

---

## Step 15: Cleanup (Optional)

Delete the compute cluster:

```bash
darwin compute delete --cluster-id $CLUSTER_ID
```

---

## Summary

In this example, you learned how to:

| Step | Action | CLI Command |
|------|--------|-------------|
| 1 | Initialize platform | `sh init-example.sh` |
| 2 | Build and deploy | `./setup.sh -y && ./start.sh` |
| 3 | Configure CLI | `darwin config set --env darwin-local` |
| 4 | Create cluster | `darwin compute create --file cluster-config.yaml` |
| 5 | Access Jupyter | Browser: `http://localhost/kind-0/{cluster_id}-jupyter/lab` |
| 6 | Train model | Run notebook cells |
| 7 | Verify model | `darwin mlflow model get --name IrisSparkRFClassifier` |
| 8 | Stop cluster | `darwin compute stop --cluster-id $CLUSTER_ID` |
| 9 | Create serve | `darwin serve create --name iris-spark-classifier ...` |
| 10 | Configure serve | `darwin serve config create ...` |
| 11 | Build artifact | `darwin serve artifact create ...` |
| 12 | Deploy model | `darwin serve deploy-model ...` |
| 13 | Test inference | `curl -X POST .../predict` |
| 14 | Undeploy | `darwin serve undeploy-model ...` |

---

## Troubleshooting

### Cluster not starting

```bash
# Check cluster manager logs
kubectl logs -n darwin -l app=darwin-cluster-manager

# Check compute service logs
kubectl logs -n darwin -l app=darwin-compute
```

### MLflow connection issues

```bash
# Verify MLflow service is running
kubectl get pods -n darwin -l app=darwin-mlflow-lib

# Check MLflow app logs
kubectl logs -n darwin -l app=darwin-mlflow-app
```

### Serve deployment failing

```bash
# Check artifact builder status
darwin serve artifact jobs

# Check ml-serve-app logs
kubectl logs -n darwin -l app=ml-serve-app
```

### Port forwarding issues

```bash
# Restart ingress
kubectl rollout restart deployment -n ingress-nginx ingress-nginx-controller
```

---

## Files in This Example

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `train_iris_model_spark.ipynb` | Complete training notebook |
| `train_iris_model.ipynb` | Alternative non-Spark version |
| `init-example.sh` | Quick setup script |
| `cluster-config.yaml` | Compute cluster configuration |
| `serve-config.yaml` | ML-Serve infrastructure config |
| `sample-request.json` | Sample inference request |
