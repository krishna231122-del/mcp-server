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

"""Lifecycle and deployment tools for KServe InferenceServices."""

import logging
from typing import Any

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import (
    PreviewResponse,
    ToolError,
    ToolResponse,
    exception_details,
    is_k8s_not_found,
)
from kubeflow_mcp.core.policy import get_effective_persona
from kubeflow_mcp.core.security import check_namespace_allowed, validate_k8s_name
from kubeflow_mcp.serving.api.discovery import get_client

logger = logging.getLogger(__name__)


def _require_mcp_ownership(name: str, namespace: str, hint: str) -> ToolError | None:
    """Block non-admin personas from mutating InferenceServices MCP did not create."""
    if get_effective_persona() == "platform-admin":
        return None

    ownership = mcp_utils.get_kserve_ownership(name, namespace)
    if ownership is None:
        return ToolError(
            error=f"Cannot verify ownership of InferenceService '{name}' (API error)",
            error_code=ErrorCode.SDK_ERROR,
            details={"hint": "Retry, or use platform-admin persona to bypass."},
        )
    if ownership == mcp_utils.OWNERSHIP_MISSING:
        return ToolError(
            error=f"InferenceService '{name}' not found",
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
        )
    if ownership == mcp_utils.OWNERSHIP_UNMANAGED:
        return ToolError(
            error=f"InferenceService '{name}' was not created by MCP",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"hint": hint},
        )
    return None


def deploy_model(
    name: str,
    storage_uri: str,
    model_format: str = "sklearn",
    namespace: str | None = None,
    cpu_limit: str = "1",
    memory_limit: str = "2Gi",
    gpu_count: int | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Deploy a model artifact as a KServe InferenceService.

    Requires two-phase confirmation (confirmed=True) to apply changes.
    First call returns a preview of the manifest.

    Args:
        name: Name of the InferenceService.
        storage_uri: Model artifact URI (e.g., gs://..., s3://..., pvc://...).
        model_format: Model framework format (e.g., sklearn, xgboost, pytorch, triton).
        namespace: K8s namespace. Uses "default" when omitted.
        cpu_limit: CPU resource limit (default: 1).
        memory_limit: Memory resource limit (default: 2Gi).
        gpu_count: Optional GPU count.
        confirmed: Set True to deploy. False returns a preview.

    Returns:
        dict: PreviewResponse when confirmed=False, or ToolResponse on deployment.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    ns = namespace or "default"

    resources: dict[str, Any] = {
        "limits": {"cpu": cpu_limit, "memory": memory_limit},
        "requests": {"cpu": "100m", "memory": "256Mi"},
    }
    if gpu_count and gpu_count > 0:
        resources["limits"]["nvidia.com/gpu"] = str(gpu_count)

    manifest: dict[str, Any] = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": ns,
            "labels": {
                mcp_utils.MCP_MANAGED_LABEL: mcp_utils.MCP_MANAGED_VALUE,
            },
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
        return PreviewResponse(
            message=f"Confirmation required to deploy InferenceService '{name}'.",
            config=manifest,
        ).model_dump()

    try:
        client = get_client()
        created = client.create_inference_service(manifest=manifest, namespace=ns)
        return ToolResponse(
            data={
                "status": "DEPLOYED",
                "name": name,
                "namespace": ns,
                "uid": created.get("metadata", {}).get("uid"),
                "message": f"InferenceService '{name}' deployed successfully",
            }
        ).model_dump()
    except Exception as e:
        logger.warning("deploy_model(%s) failed: %s", name, e, exc_info=True)
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def delete_inference_service(
    name: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Delete a KServe InferenceService permanently.

    This operation is irreversible - all predictor resources are removed.
    Requires confirmed=True to execute. First call returns a preview.

    Args:
        name: InferenceService name to delete.
        namespace: K8s namespace. Uses "default" when omitted.
        confirmed: Set True to delete. False returns a preview.

    Returns:
        dict: PreviewResponse when confirmed=False, or ToolResponse on deletion.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    ns = namespace or "default"

    owner_err = _require_mcp_ownership(
        name,
        ns,
        "Data scientists can only delete InferenceServices created through MCP tools. "
        "Use platform-admin persona to delete externally created services.",
    )
    if owner_err is not None:
        return owner_err.model_dump()

    if not confirmed:
        return PreviewResponse(
            message=f"Will permanently delete InferenceService '{name}'. Set confirmed=True to proceed.",
            config={"service": name, "namespace": ns},
        ).model_dump()

    try:
        client = get_client()
        client.delete_inference_service(name=name, namespace=ns)
        return ToolResponse(
            data={
                "service": name,
                "namespace": ns,
                "deleted": True,
                "message": f"InferenceService '{name}' deleted successfully",
            }
        ).model_dump()
    except Exception as e:
        logger.warning("delete_inference_service(%s) failed: %s", name, e, exc_info=True)
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
