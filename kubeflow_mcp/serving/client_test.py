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

"""Unit tests for KServeClient."""

from unittest.mock import MagicMock

import pytest
from kubernetes.client.rest import ApiException

from kubeflow_mcp.serving.client import GROUP, PLURAL, VERSION, KServeClient


@pytest.fixture
def mock_apis():
    custom_api = MagicMock()
    core_api = MagicMock()
    return custom_api, core_api


@pytest.fixture
def client(mock_apis):
    custom_api, core_api = mock_apis
    return KServeClient(custom_api=custom_api, core_api=core_api)


def test_list_inference_services_success(client, mock_apis):
    custom_api, _ = mock_apis
    custom_api.list_namespaced_custom_object.return_value = {
        "items": [{"metadata": {"name": "model-a"}}]
    }

    items = client.list_inference_services(namespace="test-ns")
    assert len(items) == 1
    assert items[0]["metadata"]["name"] == "model-a"
    custom_api.list_namespaced_custom_object.assert_called_once_with(
        group=GROUP,
        version=VERSION,
        namespace="test-ns",
        plural=PLURAL,
    )


def test_list_inference_services_404_returns_empty(client, mock_apis):
    custom_api, _ = mock_apis
    custom_api.list_namespaced_custom_object.side_effect = ApiException(status=404)

    items = client.list_inference_services(namespace="missing-ns")
    assert items == []


def test_list_inference_services_api_exception_raises(client, mock_apis):
    custom_api, _ = mock_apis
    custom_api.list_namespaced_custom_object.side_effect = ApiException(status=500)

    with pytest.raises(ApiException):
        client.list_inference_services(namespace="err-ns")


def test_get_inference_service_found(client, mock_apis):
    custom_api, _ = mock_apis
    custom_api.get_namespaced_custom_object.return_value = {
        "metadata": {"name": "model-a", "namespace": "test-ns"}
    }

    result = client.get_inference_service(name="model-a", namespace="test-ns")
    assert result is not None
    assert result["metadata"]["name"] == "model-a"
    custom_api.get_namespaced_custom_object.assert_called_once_with(
        group=GROUP,
        version=VERSION,
        namespace="test-ns",
        plural=PLURAL,
        name="model-a",
    )


def test_get_inference_service_404_returns_none(client, mock_apis):
    custom_api, _ = mock_apis
    custom_api.get_namespaced_custom_object.side_effect = ApiException(status=404)

    result = client.get_inference_service(name="missing", namespace="test-ns")
    assert result is None


def test_get_inference_service_api_exception_raises(client, mock_apis):
    custom_api, _ = mock_apis
    custom_api.get_namespaced_custom_object.side_effect = ApiException(status=403)

    with pytest.raises(ApiException):
        client.get_inference_service(name="forbidden", namespace="test-ns")


def test_create_inference_service(client, mock_apis):
    custom_api, _ = mock_apis
    manifest = {"metadata": {"name": "new-model"}}
    custom_api.create_namespaced_custom_object.return_value = {
        "metadata": {"name": "new-model", "uid": "12345"}
    }

    result = client.create_inference_service(manifest=manifest, namespace="test-ns")
    assert result["metadata"]["uid"] == "12345"
    custom_api.create_namespaced_custom_object.assert_called_once_with(
        group=GROUP,
        version=VERSION,
        namespace="test-ns",
        plural=PLURAL,
        body=manifest,
    )


def test_delete_inference_service(client, mock_apis):
    custom_api, _ = mock_apis
    custom_api.delete_namespaced_custom_object.return_value = {"status": "Success"}

    result = client.delete_inference_service(name="old-model", namespace="test-ns")
    assert result["status"] == "Success"
    custom_api.delete_namespaced_custom_object.assert_called_once_with(
        group=GROUP,
        version=VERSION,
        namespace="test-ns",
        plural=PLURAL,
        name="old-model",
    )


def test_get_predictor_logs_api_exception_listing_pods(client, mock_apis):
    _, core_api = mock_apis
    core_api.list_namespaced_pod.side_effect = ApiException(status=403, reason="Forbidden")

    result = client.get_predictor_logs(name="model-a", namespace="test-ns")
    assert "Failed to list pods in namespace 'test-ns': Forbidden (403)" in result


def test_get_predictor_logs_no_pods_found(client, mock_apis):
    _, core_api = mock_apis
    mock_pod_list = MagicMock()
    mock_pod_list.items = []
    core_api.list_namespaced_pod.return_value = mock_pod_list

    result = client.get_predictor_logs(name="model-a", namespace="test-ns")
    assert "No predictor pods found for InferenceService 'model-a'" in result
    assert "scale-to-zero" in result


def test_get_predictor_logs_specific_pod_not_found(client, mock_apis):
    _, core_api = mock_apis
    pod1 = MagicMock()
    pod1.metadata.name = "pod-1"
    pod1.metadata.creation_timestamp = 100
    mock_pod_list = MagicMock()
    mock_pod_list.items = [pod1]
    core_api.list_namespaced_pod.return_value = mock_pod_list

    result = client.get_predictor_logs(
        name="model-a", namespace="test-ns", pod_name="nonexistent-pod"
    )
    assert "Pod 'nonexistent-pod' not found" in result
    assert "Available pods: ['pod-1']" in result


def test_get_predictor_logs_pod_pending(client, mock_apis):
    _, core_api = mock_apis
    pod = MagicMock()
    pod.metadata.name = "pod-pending"
    pod.metadata.creation_timestamp = 100
    pod.status.phase = "Pending"
    cs = MagicMock()
    cs.name = "kserve-container"
    cs.state.waiting.reason = "ContainerCreating"
    pod.status.init_container_statuses = []
    pod.status.container_statuses = [cs]

    mock_pod_list = MagicMock()
    mock_pod_list.items = [pod]
    core_api.list_namespaced_pod.return_value = mock_pod_list

    result = client.get_predictor_logs(name="model-a", namespace="test-ns")
    assert "Pod 'pod-pending' is still in 'Pending' state" in result
    assert "Logs are not available yet" in result


def test_get_predictor_logs_storage_initializer_failure(client, mock_apis):
    _, core_api = mock_apis
    pod = MagicMock()
    pod.metadata.name = "pod-storage-err"
    pod.metadata.creation_timestamp = 200
    pod.status.phase = "Running"

    init_cs = MagicMock()
    init_cs.name = "storage-initializer"
    init_cs.state.terminated.exit_code = 1
    pod.status.init_container_statuses = [init_cs]
    pod.status.container_statuses = []
    c1 = MagicMock()
    c1.name = "kserve-container"
    pod.spec.containers = [c1]

    mock_pod_list = MagicMock()
    mock_pod_list.items = [pod]
    core_api.list_namespaced_pod.return_value = mock_pod_list
    core_api.read_namespaced_pod_log.return_value = "storage error: bucket not found"

    result = client.get_predictor_logs(name="model-a", namespace="test-ns")
    assert "Container: storage-initializer" in result
    assert "WARNING: Detected failure in 'storage-initializer' init container" in result
    assert "storage error: bucket not found" in result
    core_api.read_namespaced_pod_log.assert_called_once_with(
        name="pod-storage-err",
        namespace="test-ns",
        container="storage-initializer",
        tail_lines=100,
    )


def test_get_predictor_logs_success(client, mock_apis):
    _, core_api = mock_apis
    pod1 = MagicMock()
    pod1.metadata.name = "pod-1"
    pod1.metadata.creation_timestamp = 100
    pod1.status.phase = "Running"
    pod1.status.init_container_statuses = []
    pod1.status.container_statuses = []
    c1 = MagicMock()
    c1.name = "kserve-container"
    pod1.spec.containers = [c1]

    pod2 = MagicMock()
    pod2.metadata.name = "pod-2"
    pod2.metadata.creation_timestamp = 200  # newer
    pod2.status.phase = "Running"
    pod2.status.init_container_statuses = []
    pod2.status.container_statuses = []
    pod2.spec.containers = [c1]

    mock_pod_list = MagicMock()
    mock_pod_list.items = [pod1, pod2]
    core_api.list_namespaced_pod.return_value = mock_pod_list
    core_api.read_namespaced_pod_log.return_value = "Serving predictions on port 8080"

    result = client.get_predictor_logs(name="model-a", namespace="test-ns")
    assert "Pod: pod-2 | Container: kserve-container" in result
    assert "Other running replicas: pod-1" in result
    assert "Serving predictions on port 8080" in result
    core_api.read_namespaced_pod_log.assert_called_once_with(
        name="pod-2",
        namespace="test-ns",
        container="kserve-container",
        tail_lines=100,
    )


def test_get_predictor_logs_read_log_api_exception(client, mock_apis):
    _, core_api = mock_apis
    pod = MagicMock()
    pod.metadata.name = "pod-1"
    pod.metadata.creation_timestamp = 100
    pod.status.phase = "Running"
    pod.status.init_container_statuses = []
    pod.status.container_statuses = []
    c1 = MagicMock()
    c1.name = "kserve-container"
    pod.spec.containers = [c1]

    mock_pod_list = MagicMock()
    mock_pod_list.items = [pod]
    core_api.list_namespaced_pod.return_value = mock_pod_list
    core_api.read_namespaced_pod_log.side_effect = ApiException(status=500, reason="Internal Error")

    result = client.get_predictor_logs(name="model-a", namespace="test-ns")
    assert (
        "Failed to fetch logs for pod 'pod-1', container 'kserve-container': Internal Error (500)"
        in result
    )
