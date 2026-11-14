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

"""Unit tests for serving discovery tools."""

from unittest.mock import MagicMock

import pytest
from kubernetes.client.rest import ApiException

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.serving.api.discovery import (
    get_inference_logs,
    get_inference_service,
    list_inference_services,
    set_client,
)


@pytest.fixture
def mock_client():
    client = MagicMock()
    set_client(client)
    yield client
    set_client(None)


def test_list_inference_services_invalid_namespace(mock_client):
    res = list_inference_services(namespace="Invalid_Namespace!")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR
    mock_client.list_inference_services.assert_not_called()


def test_list_inference_services_success(mock_client):
    mock_client.list_inference_services.return_value = [
        {
            "metadata": {"name": "sklearn-iris", "namespace": "default"},
            "status": {
                "url": "http://sklearn-iris.default.example.com",
                "conditions": [
                    {"type": "Ready", "status": "True"},
                    {"type": "PredictorReady", "status": "True"},
                ],
            },
            "spec": {
                "predictor": {
                    "model": {
                        "modelFormat": {"name": "sklearn"},
                        "storageUri": "gs://kfmodels/sklearn/iris",
                    }
                }
            },
        },
        {
            "metadata": {"name": "pending-model", "namespace": "default"},
            "status": {
                "conditions": [
                    {"type": "Ready", "status": "False"},
                ]
            },
            "spec": {},
        },
    ]

    res = list_inference_services()
    assert res["success"] is True
    data = res["data"]
    assert data["total"] == 2
    services = data["services"]
    assert services[0]["name"] == "sklearn-iris"
    assert services[0]["ready"] is True
    assert services[0]["url"] == "http://sklearn-iris.default.example.com"
    assert services[0]["model_format"] == {"name": "sklearn"}

    assert services[1]["name"] == "pending-model"
    assert services[1]["ready"] is False
    assert services[1]["url"] == "Pending"


def test_list_inference_services_error(mock_client):
    mock_client.list_inference_services.side_effect = RuntimeError("K8s API down")
    res = list_inference_services()
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.SDK_ERROR
    assert "K8s API down" in res["error"]


def test_get_inference_service_invalid_name():
    res = get_inference_service(name="Invalid_Name!")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR


def test_get_inference_service_invalid_namespace():
    res = get_inference_service(name="model-a", namespace="Bad_Namespace!")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR


def test_get_inference_service_not_found(mock_client):
    mock_client.get_inference_service.return_value = None
    res = get_inference_service(name="missing-model")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.RESOURCE_NOT_FOUND
    assert "missing-model" in res["error"]


def test_get_inference_service_success(mock_client):
    mock_client.get_inference_service.return_value = {
        "metadata": {"name": "torch-model", "namespace": "ml-team"},
        "status": {
            "url": "http://torch-model.ml-team.example.com",
            "conditions": [{"type": "Ready", "status": "True"}],
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "pytorch"},
                    "storageUri": "s3://models/torch/v1",
                    "resources": {"limits": {"cpu": "2"}},
                }
            }
        },
    }

    res = get_inference_service(name="torch-model", namespace="ml-team")
    assert res["success"] is True
    data = res["data"]
    assert data["name"] == "torch-model"
    assert data["namespace"] == "ml-team"
    assert data["url"] == "http://torch-model.ml-team.example.com"
    assert data["ready"] is True
    assert data["model_format"] == {"name": "pytorch"}
    assert data["storage_uri"] == "s3://models/torch/v1"
    assert data["resources"] == {"limits": {"cpu": "2"}}


def test_get_inference_service_404_api_exception(mock_client):
    mock_client.get_inference_service.side_effect = ApiException(status=404)
    res = get_inference_service(name="model-404")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.RESOURCE_NOT_FOUND


def test_get_inference_service_other_api_exception(mock_client):
    mock_client.get_inference_service.side_effect = ApiException(status=500, reason="Internal")
    res = get_inference_service(name="model-err")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.SDK_ERROR


def test_get_inference_logs_invalid_name():
    res = get_inference_logs(name="Bad_Name!")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR


def test_get_inference_logs_invalid_namespace():
    res = get_inference_logs(name="model-a", namespace="Bad_Namespace!")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.VALIDATION_ERROR


def test_get_inference_logs_success(mock_client):
    mock_client.get_predictor_logs.return_value = "=== Pod: p1 ===\nReady"
    res = get_inference_logs(name="model-a", namespace="test-ns", tail_lines=50)
    assert res["success"] is True
    assert res["data"]["service"] == "model-a"
    assert res["data"]["namespace"] == "test-ns"
    assert res["data"]["logs"] == "=== Pod: p1 ===\nReady"
    mock_client.get_predictor_logs.assert_called_once_with(
        name="model-a",
        namespace="test-ns",
        pod_name=None,
        container=None,
        tail_lines=50,
    )


def test_get_inference_logs_error(mock_client):
    mock_client.get_predictor_logs.side_effect = RuntimeError("Timeout")
    res = get_inference_logs(name="model-a")
    assert res["success"] is False
    assert res["error_code"] == ErrorCode.SDK_ERROR
