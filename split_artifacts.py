
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)

mlflow.set_experiment("VisionGuard_Image_Classification")

with mlflow.start_run(run_name="Dataset_Split_Artifacts"):
    mlflow.log_artifact("train.csv")
    mlflow.log_artifact("val.csv")
    mlflow.log_artifact("test.csv")

    mlflow.set_tag("stage", "dataset_split_artifacts")

    print("Successfully logged train.csv, val.csv, and test.csv!")
