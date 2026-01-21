"""Minimal MLflow connection test."""
import mlflow

# Set your MLflow URI here
MLFLOW_URI = "https://dagshub.com/TuanMinhajSeedin/Medical-Image-Segmentation.mlflow"

print(f"Testing MLflow connection: {MLFLOW_URI}")

try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Try to list experiments - this tests server connection
    experiments = mlflow.search_experiments(max_results=1)
    print("✓ Connection successful!")
    print(f"✓ Found {len(experiments)} experiment(s)")
    
    # Try to create a simple test run
    with mlflow.start_run(run_name="test") as run:
        mlflow.log_param("test", "value")
        mlflow.log_metric("test_metric", 1.0)
        print(f"✓ Test run created: {run.info.run_id}")
    
    print("\n✓ All tests passed! MLflow server is working.")
    
except Exception as e:
    print(f"✗ Connection failed: {e}")

