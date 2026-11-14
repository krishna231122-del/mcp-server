# KServe Serving Patterns

Guidance and best practices for deploying and managing models with KServe on Kubernetes.

## Overview

KServe provides serverless model serving for machine learning models on Kubernetes. It manages the lifecycle of `InferenceService` CRDs, offering features like auto-scaling (including scale-to-zero), canary rollouts, and standard protocols (v1 and v2 dataplane).

## Supported Model Formats

When deploying models via `deploy_model(model_format=...)`, common supported runtimes include:

| Format | Framework | Description |
|---|---|---|
| `sklearn` | Scikit-learn | Serves serialized pickle/joblib models |
| `xgboost` | XGBoost | Serves XGBoost booster models |
| `pytorch` | PyTorch (TorchServe) | Serves `.pt` / `.mar` files |
| `tensorflow` | TensorFlow (TFServing) | Serves SavedModel format |
| `onnx` | ONNX Runtime | Serves `.onnx` models |
| `triton` | NVIDIA Triton | Multi-framework high performance serving |
| `huggingface` | vLLM / HuggingFace | High-throughput LLM serving |

## Storage URI Schemes

KServe downloads model artifacts inside the `storage-initializer` init container before the predictor container starts:

- **Google Cloud Storage**: `gs://bucket-name/path/to/model`
- **AWS S3**: `s3://bucket-name/path/to/model`
- **Kubernetes PVC**: `pvc://pvc-name/path/to/model`
- **OCI / Registry**: `oci://registry/repo:tag`

## Workflow

1. **Deploy**:
   - Call `deploy_model(name="my-model", storage_uri="s3://bucket/model")` with `confirmed=False` to inspect the generated manifest.
   - Re-run with `confirmed=True` to create the `InferenceService`.
2. **Monitor**:
   - Call `list_inference_services()` to check status and readiness.
   - Call `get_inference_service(name="my-model")` to inspect endpoints and conditions.
   - Call `get_inference_logs(name="my-model")` to monitor runtime logs and debug storage download errors.
3. **Delete**:
   - Call `delete_inference_service(name="my-model", confirmed=True)` to remove the service.

## Troubleshooting

- **Storage Download Failure**: If the predictor pod fails during startup, check `get_inference_logs()`. KServe logs will highlight `storage-initializer` failures if credentials or bucket paths are incorrect.
- **Scale to Zero**: If no predictor pods are running, the service may be idle. Send a test inference request to the endpoint URL to trigger pod startup.
