"""KServe Serving module for kubeflow_mcp."""

from collections.abc import Callable

from kubeflow_mcp.serving.api.discovery import (
    get_inference_logs,
    list_inference_services,
)
from kubeflow_mcp.serving.api.lifecycle import deploy_model

# 1. Export list of callable tool functions
TOOLS: list[Callable] = [
    list_inference_services,
    get_inference_logs,
    deploy_model,
]

# 2. Rich descriptions for agent discovery
CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_inference_services": (
        "List all deployed KServe model inference services in a namespace, "
        "including their status, URLs, and runtime formats."
    ),
    "get_inference_logs": (
        "Retrieve runtime predictor pod logs for debugging failing or active "
        "InferenceService rollouts."
    ),
    "deploy_model": (
        "Deploy a trained model as a live KServe InferenceService. "
        "Mutating operation requiring confirmed=True to execute."
    ),
}

# 3. Optional tool annotations (metadata / hints)
CLIENT_TOOL_ANNOTATIONS: dict[str, dict[str, bool]] = {
    "list_inference_services": {"read_only": True},
    "get_inference_logs": {"read_only": True},
    "deploy_model": {"read_only": False, "requires_confirmation": True},
}
