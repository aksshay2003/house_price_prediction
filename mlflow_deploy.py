import mlflow
model_name="house_price_predictor"
model_name_prod="house_price_pred_prod"
dev_model_uri=f"models:/{model_name}@challenger"
client=mlflow.MlflowClient()
# client.copy_model_version(
#     src_mv={"name": model_name, "version": 1},
#     dst_name=model_name_prod
# )
client.copy_model_version(
    src_model_uri=dev_model_uri,
    dst_name=model_name_prod,
)
