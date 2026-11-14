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

"""Architecture tests for serving (KServe) module."""

import inspect
from unittest.mock import patch

import pytest

from kubeflow_mcp.common.constants import TOOL_TO_PHASE
from kubeflow_mcp.core.policy import DESTRUCTIVE_TOOLS, get_allowed_tools
from kubeflow_mcp.serving import (
    CLIENT_RESOURCES,
    CLIENT_TOOL_ANNOTATIONS,
    CLIENT_TOOL_DESCRIPTIONS,
    INSTRUCTION_SECTIONS,
    PHASE_TO_SECTION,
    SECTION_ORDER,
    TOOLS,
)


class TestServingMetadataConsistency:
    def test_all_tools_have_descriptions(self):
        tool_names = {t.__name__ for t in TOOLS}
        desc_names = set(CLIENT_TOOL_DESCRIPTIONS.keys())
        assert tool_names == desc_names

    def test_all_tools_have_annotations(self):
        tool_names = {t.__name__ for t in TOOLS}
        ann_names = set(CLIENT_TOOL_ANNOTATIONS.keys())
        assert tool_names == ann_names

    def test_all_tools_in_tool_phases(self):
        tool_names = {t.__name__ for t in TOOLS}
        phased_tools = set(TOOL_TO_PHASE.keys())
        orphan = tool_names - phased_tools
        assert not orphan, f"Serving TOOLS missing phase map: {orphan}"

    def test_annotation_schema(self):
        required_keys = {
            "title",
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
            "tags",
        }
        for name, ann in CLIENT_TOOL_ANNOTATIONS.items():
            missing = required_keys - set(ann.keys())
            assert not missing, f"Tool '{name}' missing annotation keys: {missing}"

    def test_destructive_tools_have_destructive_hint(self):
        for tool_name in DESTRUCTIVE_TOOLS:
            if tool_name in CLIENT_TOOL_ANNOTATIONS:
                assert CLIENT_TOOL_ANNOTATIONS[tool_name]["destructiveHint"] is True, (
                    f"Tool '{tool_name}' in DESTRUCTIVE_TOOLS but destructiveHint is False"
                )

    def test_section_order_and_instructions(self):
        assert SECTION_ORDER == ["serving"]
        assert "serving" in INSTRUCTION_SECTIONS
        assert "full" in INSTRUCTION_SECTIONS["serving"]
        assert len(PHASE_TO_SECTION) > 0
        assert len(CLIENT_RESOURCES) > 0


# Arguments each tool needs, besides namespace, to get as far as its namespace check.
_NAMESPACE_TOOL_ARGS = {
    "get_inference_service": {"name": "model-a"},
    "get_inference_logs": {"name": "model-a"},
    "deploy_model": {"name": "model-a", "storage_uri": "s3://models/a"},
    "delete_inference_service": {"name": "model-a"},
}
_NAMESPACE_TOOLS = [t for t in TOOLS if "namespace" in inspect.signature(t).parameters]


@pytest.mark.parametrize("tool", _NAMESPACE_TOOLS, ids=lambda t: t.__name__)
def test_malformed_namespace_is_rejected_before_any_api_request(tool):
    """A tool that takes a namespace must not send a malformed one to the API."""
    params = inspect.signature(tool).parameters
    args = _NAMESPACE_TOOL_ARGS.get(tool.__name__, {"name": "model-a"} if "name" in params else {})

    with (
        patch("kubernetes.config.load_kube_config"),
        patch("kubernetes.config.load_incluster_config"),
        patch("kubernetes.client.ApiClient.call_api") as call_api,
    ):
        result = tool(**args, namespace="Bad_NS!")

    assert result["error_code"] == "VALIDATION_ERROR", result
    assert "namespace" in result["error"]
    call_api.assert_not_called()


class TestServingPersonaGating:
    def test_readonly_can_discover_and_read_logs(self):
        tools = get_allowed_tools("readonly")
        assert "list_inference_services" in tools
        assert "get_inference_service" in tools
        assert "get_inference_logs" in tools

    def test_readonly_cannot_deploy_or_delete(self):
        tools = get_allowed_tools("readonly")
        assert "deploy_model" not in tools
        assert "delete_inference_service" not in tools

    def test_data_scientist_can_deploy_and_delete(self):
        tools = get_allowed_tools("data-scientist")
        assert "deploy_model" in tools
        assert "delete_inference_service" in tools

    def test_delete_inference_service_is_destructive(self):
        assert "delete_inference_service" in DESTRUCTIVE_TOOLS
