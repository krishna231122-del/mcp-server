# Copyright The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discovery tools for KServe model inference services."""

import logging
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found
from kubeflow_mcp.core.security import check_namespace_allowed, validate_k8s_name
from kubeflow_mcp.serving.client import KServeClient

logger = logging.getLogger(__name__)

_client: KServeClient | None = None


def get_client() -> KServeClient:
    """Get or create singleton KServeClient."""
    global _client
    if _client is None:
        _client = KServeClient()
    return _client


def set_client(client: KServeClient | None) -> None:
    """Set the active KServeClient (for testing)."""
    global _client
    _client = client


def list_inference_services(
    namespace: str | None = None,
) -> dict[str, Any]:
    """List deployed KServe model inference services and their status.

    Args:
        namespace: K8s namespace. Uses "default" when omitted.

    Returns:
        dict: Standard ToolResponse with list of services, status, and endpoints.
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    ns = namespace or "default"
    try:
        client = get_client()
        items = client.list_inference_services(namespace=ns)
        results = []
        for item in items:
            status = item.get("status", {})
            url = status.get("url", "Pending")
            conditions = {
                c.get("type"): c.get("status")
                for c in status.get("conditions", [])
                if isinstance(c, dict) and "type" in c
            }
            spec_predictor = item.get("spec", {}).get("predictor", {})
            model = spec_predictor.get("model", {})
            results.append(
                {
                    "name": item.get("metadata", {}).get("name"),
                    "namespace": item.get("metadata", {}).get("namespace", ns),
                    "url": url,
                    "ready": conditions.get("Ready") == "True",
                    "model_format": model.get("modelFormat", {}),
                    "conditions": conditions,
                }
            )
        return ToolResponse(
            data={
                "services": results,
                "total": len(results),
            }
        ).model_dump()
    except Exception as e:
        logger.warning("list_inference_services failed: %s", e, exc_info=True)
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_inference_service(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get details of a specific KServe InferenceService.

    Args:
        name: Name of the InferenceService.
        namespace: K8s namespace. Uses "default" when omitted.

    Returns:
        dict: Standard ToolResponse with detailed service configuration and status.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    ns = namespace or "default"
    try:
        client = get_client()
        service = client.get_inference_service(name=name, namespace=ns)
        if service is None:
            return ToolError(
                error=f"InferenceService '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()

        status = service.get("status", {})
        url = status.get("url", "Pending")
        conditions = {
            c.get("type"): c.get("status")
            for c in status.get("conditions", [])
            if isinstance(c, dict) and "type" in c
        }
        spec_predictor = service.get("spec", {}).get("predictor", {})
        model = spec_predictor.get("model", {})

        return ToolResponse(
            data={
                "name": name,
                "namespace": ns,
                "url": url,
                "ready": conditions.get("Ready") == "True",
                "model_format": model.get("modelFormat", {}),
                "storage_uri": model.get("storageUri"),
                "resources": model.get("resources", {}),
                "conditions": conditions,
                "raw_status": status,
            }
        ).model_dump()
    except Exception as e:
        logger.warning("get_inference_service(%s) failed: %s", name, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"InferenceService '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_inference_logs(
    name: str,
    namespace: str | None = None,
    pod_name: str | None = None,
    container: str | None = None,
    tail_lines: int = 100,
) -> dict[str, Any]:
    """Retrieve runtime predictor pod logs for debugging an InferenceService.

    Args:
        name: Name of the InferenceService.
        namespace: K8s namespace. Uses "default" when omitted.
        pod_name: Specific predictor pod name (defaults to newest).
        container: Specific container name ('kserve-container', 'storage-initializer').
        tail_lines: Number of lines to tail (default 100).

    Returns:
        dict: Standard ToolResponse containing the log text and service metadata.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    ns = namespace or "default"
    try:
        client = get_client()
        logs = client.get_predictor_logs(
            name=name,
            namespace=ns,
            pod_name=pod_name,
            container=container,
            tail_lines=tail_lines,
        )
        return ToolResponse(
            data={
                "service": name,
                "namespace": ns,
                "logs": logs,
            }
        ).model_dump()
    except Exception as e:
        logger.warning("get_inference_logs(%s) failed: %s", name, e, exc_info=True)
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
