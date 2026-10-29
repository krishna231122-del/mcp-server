"""Discovery tools for KServe model inference services."""

from typing import Any

from kubeflow_mcp.serving.client import KServeClient

_client: KServeClient | None = None


def get_client() -> KServeClient:
    global _client
    if _client is None:
        _client = KServeClient()
    return _client


def list_inference_services(namespace: str = "default") -> list[dict[str, Any]]:
    """List deployed model inference services and their URL endpoints."""
    client = get_client()
    items = client.list_inference_services(namespace=namespace)
    results = []
    for item in items:
        status = item.get("status", {})
        url = status.get("url", "Pending")
        conditions = {c["type"]: c["status"] for c in status.get("conditions", [])}
        results.append(
            {
                "name": item["metadata"]["name"],
                "namespace": item["metadata"]["namespace"],
                "url": url,
                "ready": conditions.get("Ready") == "True",
                "model_format": item.get("spec", {})
                .get("predictor", {})
                .get("model", {})
                .get("modelFormat", {}),
            }
        )
    return results


def get_inference_logs(name: str, namespace: str = "default", tail_lines: int = 100) -> str:
    """Retrieve runtime logs from a model predictor pod."""
    client = get_client()
    return client.get_predictor_logs(name=name, namespace=namespace, tail_lines=tail_lines)
