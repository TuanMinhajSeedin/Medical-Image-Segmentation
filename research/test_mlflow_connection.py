"""Simple script to test MLflow remote server connection."""
import mlflow
import os
from urllib.parse import urlparse


def test_mlflow_connection(mlflow_uri: str, username: str = None, password: str = None):
    """Test if MLflow remote server is accessible."""
    print(f"Testing MLflow connection to: {mlflow_uri}")
    print("-" * 60)
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Set credentials if provided
        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
        if password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        
        # Get tracking URI to verify it was set
        actual_uri = mlflow.get_tracking_uri()
        print(f"✓ Tracking URI set: {actual_uri}")
        
        # Check URI scheme
        parsed_uri = urlparse(actual_uri)
        print(f"✓ URI scheme: {parsed_uri.scheme}")
        print(f"✓ URI host: {parsed_uri.netloc}")
        
        # Try to list experiments (this tests server connection)
        print("\nAttempting to connect to server...")
        experiments = mlflow.search_experiments(max_results=5)
        print(f"✓ Successfully connected to MLflow server!")
        print(f"✓ Found {len(experiments)} experiment(s)")
        
        if experiments:
            print("\nRecent experiments:")
            for exp in experiments[:5]:
                print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Try to create a test run
        print("\nTesting run creation...")
        with mlflow.start_run(run_name="connection_test") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            print(f"✓ Successfully created test run: {run.info.run_id}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! MLflow server is working correctly.")
        print("=" * 60)
        return True
        
    except mlflow.exceptions.MlflowException as e:
        print(f"\n✗ MLflow error: {e}")
        if "403" in str(e):
            print("\n⚠ Authentication failed! You need to provide credentials.")
            print("  Set environment variables:")
            print("  - MLFLOW_TRACKING_USERNAME")
            print("  - MLFLOW_TRACKING_PASSWORD")
        print("\n" + "=" * 60)
        print("✗ MLflow server connection failed!")
        print("=" * 60)
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("\n" + "=" * 60)
        print("✗ MLflow server connection failed!")
        print("=" * 60)
        return False


if __name__ == "__main__":
    # Get MLflow URI from config
    from LiverTumorSegmentation.config.configuration import ConfigurationManager
    
    cm = ConfigurationManager()
    eval_section = getattr(cm.config, "evaluation", None)
    
    if eval_section and hasattr(eval_section, "mlflow_uri"):
        mlflow_uri = eval_section.mlflow_uri
    else:
        mlflow_uri = input("Enter MLflow tracking URI: ").strip()
        if not mlflow_uri:
            print("No MLflow URI provided. Exiting.")
            exit(1)
    
    # Check for credentials in environment variables or prompt user
    username = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
    password = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
    
    if not username:
        username = input("Enter DagsHub username (or press Enter to skip): ").strip() or None
    if not password and username:
        password = input("Enter DagsHub password/token: ").strip() or None
    
    test_mlflow_connection(mlflow_uri, username, password)



