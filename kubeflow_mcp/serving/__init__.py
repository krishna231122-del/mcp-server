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

"""KServe Serving module for kubeflow_mcp."""

from collections.abc import Callable
from typing import Any

from kubeflow_mcp.serving.api.discovery import (
    get_inference_logs,
    get_inference_service,
    list_inference_services,
)
from kubeflow_mcp.serving.api.lifecycle import (
    delete_inference_service,
    deploy_model,
)

MODULE_INFO = {
    "name": "serving",
    "description": "Model serving and inference management with KServe",
    "sdk_client": "kubeflow_mcp.serving.client.KServeClient",
    "status": "implemented",
}

# 1. Export list of callable tool functions
TOOLS: list[Callable[..., Any]] = [
    list_inference_services,
    get_inference_service,
    get_inference_logs,
    deploy_model,
    delete_inference_service,
]

# 2. Rich descriptions for agent discovery
CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_inference_services": (
        "List all deployed KServe model inference services in a namespace, "
        "including their status, URLs, and runtime formats."
    ),
    "get_inference_service": (
        "Get details of a specific InferenceService including URL, conditions, "
        "model format, and storage URI."
    ),
    "get_inference_logs": (
        "Retrieve runtime predictor pod logs for debugging failing or active "
        "InferenceService rollouts."
    ),
    "deploy_model": (
        "Deploy a trained model as a live KServe InferenceService. "
        "Mutating operation requiring confirmed=True to execute."
    ),
    "delete_inference_service": (
        "[DESTRUCTIVE] Delete a KServe InferenceService permanently. "
        "Requires confirmed=True to execute."
    ),
}

# 3. Tool annotations (metadata / hints)
CLIENT_TOOL_ANNOTATIONS: dict[str, dict[str, Any]] = {
    "list_inference_services": {
        "title": "List Inference Services",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "serving", "inferenceservice"],
    },
    "get_inference_service": {
        "title": "Get Inference Service Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "serving", "inferenceservice"],
    },
    "get_inference_logs": {
        "title": "Get Inference Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "serving", "logs", "debug"],
    },
    "deploy_model": {
        "title": "Deploy Model",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["lifecycle", "serving", "deploy"],
    },
    "delete_inference_service": {
        "title": "Delete Inference Service",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "serving", "cleanup"],
    },
}

# 4. Resources owned by serving
CLIENT_RESOURCES: dict[str, tuple[str, str]] = {
    "serving://guides/serving-patterns": (
        "resources/serving-patterns.md",
        "KServe model serving architectures and patterns.",
    ),
}

# 5. Instruction sections
INSTRUCTION_SECTIONS: dict[str, dict[str, str]] = {
    "serving": {
        "full": """\
MODEL SERVING (KServe):
- list_inference_services() -> list deployed model inference services, URLs, and readiness
- get_inference_service(name) -> inspect service spec, model format, conditions, and endpoint URL
- get_inference_logs(name) -> view predictor pod runtime logs (intelligently targets kserve-container or storage-initializer)
- deploy_model(name, storage_uri, ...) -> deploy model artifact as InferenceService (preview first, confirmed=True to execute)
- delete_inference_service(name, confirmed=True) -> permanently remove InferenceService (preview first)""",
    }
}

PHASE_TO_SECTION: dict[str, str | None] = {
    "planning": None,
    "discovery": "serving",
    "training": None,
    "monitoring": "serving",
    "lifecycle": "serving",
    "platform": None,
    "health": None,
}

SECTION_ORDER: list[str] = ["serving"]

__all__ = [
    "MODULE_INFO",
    "TOOLS",
    "CLIENT_TOOL_DESCRIPTIONS",
    "CLIENT_TOOL_ANNOTATIONS",
    "CLIENT_RESOURCES",
    "INSTRUCTION_SECTIONS",
    "SECTION_ORDER",
    "PHASE_TO_SECTION",
    *[t.__name__ for t in TOOLS],
]
