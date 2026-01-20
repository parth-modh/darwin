# Diabetes Regression with XGBoost on Spark - End-to-End ML Platform Example

This example demonstrates the complete ML lifecycle on the Darwin platform using distributed XGBoost with Spark for predicting diabetes disease progression.

## Overview

You will learn how to:
1. Set up the Darwin ML platform with required services
2. Create and manage a compute cluster with Spark support
3. Train an XGBoost regression model with distributed Spark training
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

## Dataset

The Diabetes dataset contains 442 samples with 10 baseline variables to predict disease progression one year after baseline:

| Feature | Description |
|---------|-------------|
| age | Age in years (normalized) |
| sex | Sex (normalized) |
| bmi | Body mass index (normalized) |
| bp | Average blood pressure (normalized) |
| s1 | Total serum cholesterol (normalized) |
| s2 | Low-density lipoproteins (normalized) |
| s3 | High-density lipoproteins (normalized) |
| s4 | Total cholesterol / HDL (normalized) |
| s5 | Log of serum triglycerides level (normalized) |
| s6 | Blood sugar level (normalized) |

**Target**: Quantitative measure of disease progression (continuous value)

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
sh examples/xgboost-diabetes-regression/init-example.sh
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
darwin compute create --file examples/xgboost-diabetes-regression/cluster-config.yaml
```

Expected output:
```
Cluster created successfully!
Cluster ID: <CLUSTER_ID>
Name: diabetes-xgboost-spark-example
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

1. Create a new Python 3 notebook or upload `train_xgboost_diabetes_spark.ipynb`

2. If creating a new notebook, copy the cells from `train_xgboost_diabetes_spark.ipynb`:

**Cell 1: Install Dependencies**
```python
# Fix pyOpenSSL/cryptography compatibility issue first
%pip install --upgrade pyOpenSSL cryptography

# Install main dependencies
%pip install xgboost>=2.0.0 pandas numpy scikit-learn mlflow pyspark
```

**Cell 2: Import Libraries**
```python
import os
import json
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime

# XGBoost imports
import xgboost as xgb
from xgboost.spark import SparkXGBRegressor

# Spark imports
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# MLflow imports
import mlflow
import mlflow.xgboost
from mlflow import set_tracking_uri, set_experiment
from mlflow.client import MlflowClient

# Scikit-learn imports
from sklearn.datasets import load_diabetes

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
    "spark.sql.shuffle.partitions": "10",
    "spark.default.parallelism": "10",
}

ray.init()
spark = init_spark_with_configs(spark_configs=spark_configs)
print(f"Spark version: {spark.version}")

# Install XGBoost on executors
def install_packages(iterator):
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "xgboost>=2.0.0", "scikit-learn", "pandas", "numpy",
                          "-q", "--disable-pip-version-check"])
    yield True

num_executors = spark.sparkContext.defaultParallelism
spark.sparkContext.parallelize(range(num_executors), num_executors) \
    .mapPartitions(install_packages).collect()
print(f"XGBoost installed on {num_executors} executors")
```

**Cell 4: Setup MLflow**
```python
MLFLOW_URI = "http://darwin-mlflow-lib.darwin.svc.cluster.local:8080"
USERNAME = "abc@gmail.com"
PASSWORD = "password"
EXPERIMENT_NAME = "diabetes_xgboost_spark_regression"
MODEL_NAME = "DiabetesXGBoostSparkRegressor"

os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD
set_tracking_uri(MLFLOW_URI)
client = MlflowClient(MLFLOW_URI)
set_experiment(experiment_name=EXPERIMENT_NAME)
print(f"MLflow configured: {MLFLOW_URI}")
```

**Cell 5: Load and Prepare Data**
```python
# Load Diabetes dataset
data = load_diabetes(as_frame=True)
pdf = data.data.copy()
pdf['target'] = data.target

# Convert to Spark DataFrame
df = spark.createDataFrame(pdf)
feature_cols = [c for c in df.columns if c != 'target']

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train samples: {train_df.count()}, Test samples: {test_df.count()}")
print(f"Target range: {pdf['target'].min():.1f} - {pdf['target'].max():.1f}")
```

**Cell 6: Train Model**
```python
# Define hyperparameters
hyperparams = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# Create SparkXGBRegressor
xgb_regressor = SparkXGBRegressor(
    features_col="features",
    label_col="target",
    prediction_col="prediction",
    objective="reg:squarederror",
    max_depth=hyperparams["max_depth"],
    learning_rate=hyperparams["learning_rate"],
    n_estimators=hyperparams["n_estimators"],
    subsample=hyperparams["subsample"],
    colsample_bytree=hyperparams["colsample_bytree"],
    random_state=hyperparams["random_state"],
)

with mlflow.start_run(run_name=f"spark_xgboost_diabetes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Train model (distributed)
    model = xgb_regressor.fit(train_df)
    print("Model trained!")
    
    # Evaluate
    test_pred = model.transform(test_df)
    evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction")
    rmse = evaluator.setMetricName("rmse").evaluate(test_pred)
    mae = evaluator.setMetricName("mae").evaluate(test_pred)
    r2 = evaluator.setMetricName("r2").evaluate(test_pred)
    
    # Log to MLflow
    mlflow.log_params(hyperparams)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    
    # Save model artifacts
    native_model = model.get_booster()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = os.path.join(tmpdir, "model.json")
        native_model.save_model(model_file)
        mlflow.log_artifact(model_file, artifact_path="model")
    
    run_id = mlflow.active_run().info.run_id
    experiment_id = mlflow.active_run().info.experiment_id
    
    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2: {r2:.4f}")
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

# Get details of the diabetes model
darwin mlflow model get --name DiabetesXGBoostSparkRegressor

# Get specific version details
darwin mlflow model get --name DiabetesXGBoostSparkRegressor --version 1
```

Expected output:
```
Model: DiabetesXGBoostSparkRegressor
Latest Version: 1
Description: Diabetes XGBoost Regressor
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
  --name diabetes-xgboost-regressor \
  --type api \
  --space ml-examples \
  --description "Diabetes XGBoost Spark Regressor"
```

---

## Step 10: Configure Serve Infrastructure

Create the infrastructure configuration for the serve:

```bash
darwin serve config create \
  --serve-name diabetes-xgboost-regressor \
  --env darwin-local \
  --file examples/xgboost-diabetes-regression/serve-config.yaml
```

Or configure inline:

```bash
darwin serve config create \
  --serve-name diabetes-xgboost-regressor \
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
  --serve-name diabetes-xgboost-regressor \
  --version v1.0.0 \
  --github-repo-url https://github.com/your-org/diabetes-serve-repo \
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
  --serve-name diabetes-xgboost-regressor \
  --artifact-version v1.0.0 \
  --model-uri models:/DiabetesXGBoostSparkRegressor/1 \
  --env darwin-local \
  --cores 2 \
  --memory 4 \
  --node-capacity ondemand \
  --min-replicas 1 \
  --max-replicas 3
```

Check deployment status:

```bash
darwin serve status --name diabetes-xgboost-regressor --env darwin-local
```

Wait until the status shows `RUNNING`.

---

## Step 13: Test Inference

Test the deployed model with sample requests:

**Using curl:**

```bash
curl -X POST http://localhost/serve/diabetes-xgboost-regressor/predict \
  -H "Content-Type: application/json" \
  -d @examples/xgboost-diabetes-regression/sample-request.json
```

**Sample request payload:**

```json
{
  "instances": [
    {
      "age": 0.0380759064,
      "sex": 0.0506801187,
      "bmi": 0.0616962065,
      "bp": 0.0218723550,
      "s1": -0.0442234984,
      "s2": -0.0348207628,
      "s3": -0.0434008457,
      "s4": -0.0025945987,
      "s5": 0.0199084209,
      "s6": -0.0176461252
    }
  ]
}
```

**Expected response (regression):**

```json
{
  "predictions": [
    {
      "predicted_progression": 151.34
    }
  ]
}
```

**Test with different patient samples:**

```bash
# Lower risk patient
curl -X POST http://localhost/serve/diabetes-xgboost-regressor/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "age": -0.0018820165,
      "sex": -0.0446416365,
      "bmi": -0.0514740612,
      "bp": -0.0263278347,
      "s1": -0.0084487241,
      "s2": -0.0191633397,
      "s3": 0.0744115640,
      "s4": -0.0394933829,
      "s5": -0.0683297436,
      "s6": -0.0922040496
    }]
  }'

# Higher risk patient
curl -X POST http://localhost/serve/diabetes-xgboost-regressor/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "age": 0.0852989063,
      "sex": 0.0506801187,
      "bmi": 0.0444512133,
      "bp": -0.0056671299,
      "s1": -0.0455994513,
      "s2": -0.0341944659,
      "s3": -0.0324559464,
      "s4": -0.0025945987,
      "s5": 0.0286297373,
      "s6": -0.0259303389
    }]
  }'
```

---

## Step 14: Undeploy the Serve Application

When done, undeploy the serve application:

```bash
darwin serve undeploy-model --serve-name diabetes-xgboost-regressor --env darwin-local
```

Verify undeployment:

```bash
darwin serve status --name diabetes-xgboost-regressor --env darwin-local
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
| 7 | Verify model | `darwin mlflow model get --name DiabetesXGBoostSparkRegressor` |
| 8 | Stop cluster | `darwin compute stop --cluster-id $CLUSTER_ID` |
| 9 | Create serve | `darwin serve create --name diabetes-xgboost-regressor ...` |
| 10 | Configure serve | `darwin serve config create ...` |
| 11 | Build artifact | `darwin serve artifact create ...` |
| 12 | Deploy model | `darwin serve deploy-model ...` |
| 13 | Test inference | `curl -X POST .../predict` |
| 14 | Undeploy | `darwin serve undeploy-model ...` |

---

## Comparison: Regression vs Classification Examples

| Aspect | This Example (XGBoost Regression) | Iris/Wine (Classification) |
|--------|-----------------------------------|---------------------------|
| Task Type | Regression | Multi-class Classification |
| Algorithm | SparkXGBRegressor | Random Forest / LightGBM |
| Target | Continuous (disease progression) | Categorical (species/class) |
| Metrics | RMSE, MAE, R2 | Accuracy, F1, Precision |
| Output | Single numeric value | Class + probabilities |
| Dataset | Diabetes (442, 10 features) | Iris/Wine (150-178, 4-13 features) |

---

## Regression Metrics Explained

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **RMSE** | Root Mean Square Error - average prediction error magnitude | Lower is better |
| **MAE** | Mean Absolute Error - average absolute difference | Lower is better |
| **R2** | R-squared - proportion of variance explained | Closer to 1.0 is better |

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

### XGBoost import errors

If you see XGBoost import errors in the notebook:

```python
# Install XGBoost with pip
%pip install xgboost>=2.0.0 --upgrade
```

### SparkXGBRegressor not found

Ensure you have XGBoost 2.0+ which includes Spark integration:

```python
import xgboost
print(xgboost.__version__)  # Should be >= 2.0.0
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
| `train_xgboost_diabetes_spark.ipynb` | Complete training notebook |
| `train_xgboost_diabetes.ipynb` | Alternative non-Spark version |
| `init-example.sh` | Quick setup script |
| `cluster-config.yaml` | Compute cluster configuration |
| `serve-config.yaml` | ML-Serve infrastructure config |
| `sample-request.json` | Sample inference request |
