
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)

client = mlflow.MlflowClient()

model_name = "VisionGuard_MobileNetV2"
version = "2"

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=version
)

print(f"Alias 'champion' assigned to {model_name} version {version}.")

# Verify the alias
champion = client.get_model_version_by_alias(
    name=model_name,
    alias="champion"
)

print("Verified model:", champion.name)
print("Verified version:", champion.version)
