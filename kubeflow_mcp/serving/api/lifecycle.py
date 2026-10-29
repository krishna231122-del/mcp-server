"""Lifecycle and deployment tools for KServe InferenceServices."""

from typing import Any

from kubeflow_mcp.serving.api.discovery import get_client


def deploy_model(
    name: str,
    storage_uri: str,
    model_format: str = "sklearn",
    namespace: str = "default",
    cpu_limit: str = "1",
    memory_limit: str = "2Gi",
    gpu_count: int | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Deploy a model artifact as a KServe InferenceService.

    Requires two-phase confirmation (confirmed=True) to apply changes.
    """
    resources: dict[str, Any] = {
        "limits": {"cpu": cpu_limit, "memory": memory_limit},
        "requests": {"cpu": "100m", "memory": "256Mi"},
    }
    if gpu_count and gpu_count > 0:
        resources["limits"]["nvidia.com/gpu"] = str(gpu_count)

    manifest = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": model_format},
                    "storageUri": storage_uri,
                    "resources": resources,
                }
            }
        },
    }

    # Two-phase confirmation gate
    if not confirmed:
        return {
            "status": "PREVIEW_REQUIRED",
            "message": f"Confirmation required to deploy InferenceService '{name}'.",
            "preview_manifest": manifest,
            "next_step": "Call deploy_model again with confirmed=True to execute.",
        }

    client = get_client()
    created = client.create_inference_service(manifest=manifest, namespace=namespace)
    return {
        "status": "DEPLOYED",
        "name": name,
        "namespace": namespace,
        "uid": created.get("metadata", {}).get("uid"),
    }
