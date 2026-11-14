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

"""Unit tests for serving lifecycle tools."""

from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.rest import ApiException

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.core.policy import set_effective_persona
from kubeflow_mcp.serving.api.discovery import set_client
from kubeflow_mcp.serving.api.lifecycle import delete_inference_service, deploy_model


@pytest.fixture
def mock_client():
    client = MagicMock()
    set_client(client)
    yield client
    set_client(None)


@pytest.fixture(autouse=True)
def reset_persona():
    set_effective_persona("platform-admin")
    yield
    set_effective_persona("readonly")


def test_deploy_model_invalid_name():
    res = deploy_model(name="Invalid_Name!", storage_uri="s3://models/a")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR


def test_deploy_model_invalid_namespace():
    res = deploy_model(name="model-a", storage_uri="s3://models/a", namespace="Bad_Namespace!")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR


def test_deploy_model_preview_default():
    res = deploy_model(
        name="iris-model",
        storage_uri="gs://kfmodels/iris",
        model_format="sklearn",
        namespace="ml-prod",
        cpu_limit="2",
        memory_limit="4Gi",
        gpu_count=1,
        confirmed=False,
    )
    assert res["success"] is True
    assert res["status"] == "preview"
    manifest = res["config"]
    assert manifest["metadata"]["name"] == "iris-model"
    assert manifest["metadata"]["namespace"] == "ml-prod"
    assert (
        manifest["metadata"]["labels"][mcp_utils.MCP_MANAGED_LABEL] == mcp_utils.MCP_MANAGED_VALUE
    )
    predictor = manifest["spec"]["predictor"]["model"]
    assert predictor["modelFormat"]["name"] == "sklearn"
    assert predictor["storageUri"] == "gs://kfmodels/iris"
    assert predictor["resources"]["limits"]["cpu"] == "2"
    assert predictor["resources"]["limits"]["memory"] == "4Gi"
    assert predictor["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_deploy_model_confirmed_success(mock_client):
    mock_client.create_inference_service.return_value = {
        "metadata": {"name": "iris-model", "uid": "abc-123"}
    }

    res = deploy_model(
        name="iris-model",
        storage_uri="gs://kfmodels/iris",
        confirmed=True,
    )
    assert res["success"] is True
    data = res["data"]
    assert data["status"] == "DEPLOYED"
    assert data["name"] == "iris-model"
    assert data["uid"] == "abc-123"
    mock_client.create_inference_service.assert_called_once()


def test_deploy_model_confirmed_error(mock_client):
    mock_client.create_inference_service.side_effect = RuntimeError("Webhook denied")
    res = deploy_model(name="iris-model", storage_uri="gs://kfmodels/iris", confirmed=True)
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.SDK_ERROR
    assert "Webhook denied" in res["error"]


def test_delete_inference_service_invalid_name():
    res = delete_inference_service(name="Bad_Name!")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR


def test_delete_inference_service_invalid_namespace():
    res = delete_inference_service(name="model-a", namespace="Bad_Namespace!")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR


def test_delete_inference_service_ownership_missing():
    set_effective_persona("data-scientist")
    with patch(
        "kubeflow_mcp.common.utils.get_kserve_ownership",
        return_value=mcp_utils.OWNERSHIP_MISSING,
    ):
        res = delete_inference_service(name="missing-service")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.RESOURCE_NOT_FOUND


def test_delete_inference_service_ownership_unmanaged():
    set_effective_persona("data-scientist")
    with patch(
        "kubeflow_mcp.common.utils.get_kserve_ownership",
        return_value=mcp_utils.OWNERSHIP_UNMANAGED,
    ):
        res = delete_inference_service(name="unmanaged-service")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR
    assert "not created by MCP" in res["error"]


def test_delete_inference_service_preview():
    res = delete_inference_service(name="iris-model", namespace="test-ns", confirmed=False)
    assert res["success"] is True
    assert res["status"] == "preview"
    assert res["config"]["service"] == "iris-model"
    assert res["config"]["namespace"] == "test-ns"


def test_delete_inference_service_confirmed_success(mock_client):
    res = delete_inference_service(name="iris-model", namespace="test-ns", confirmed=True)
    assert res["success"] is True
    assert res["data"]["deleted"] is True
    assert res["data"]["service"] == "iris-model"
    mock_client.delete_inference_service.assert_called_once_with(
        name="iris-model", namespace="test-ns"
    )


def test_delete_inference_service_404_error(mock_client):
    mock_client.delete_inference_service.side_effect = ApiException(status=404)
    res = delete_inference_service(name="iris-model", confirmed=True)
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.RESOURCE_NOT_FOUND


def test_delete_inference_service_other_error(mock_client):
    mock_client.delete_inference_service.side_effect = RuntimeError("RBAC error")
    res = delete_inference_service(name="iris-model", confirmed=True)
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.SDK_ERROR
