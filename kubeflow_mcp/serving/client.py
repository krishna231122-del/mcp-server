"""KServe Kubernetes client wrapper for kubeflow_mcp."""

from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException

GROUP = "serving.kserve.io"
VERSION = "v1beta1"
PLURAL = "inferenceservices"


class KServeClient:
    """Client for managing KServe InferenceService CRDs."""

    def __init__(self, kubeconfig_path: str | None = None):
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config(config_file=kubeconfig_path)

        self.custom_api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api()

    def list_inference_services(self, namespace: str = "default") -> list[dict[str, Any]]:
        """List all InferenceServices in a namespace."""
        try:
            res = self.custom_api.list_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
            )
            return res.get("items", [])
        except ApiException as e:
            if e.status == 404:
                return []
            raise

    def get_inference_service(self, name: str, namespace: str = "default") -> dict[str, Any] | None:
        """Get a single InferenceService."""
        try:
            return self.custom_api.get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=namespace,
                plural=PLURAL,
                name=name,
            )
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def create_inference_service(
        self, manifest: dict[str, Any], namespace: str = "default"
    ) -> dict[str, Any]:
        """Create a new InferenceService."""
        return self.custom_api.create_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=namespace,
            plural=PLURAL,
            body=manifest,
        )

    @staticmethod
    def _find_target_pod(
        sorted_pods: list[Any], pod_name: str | None
    ) -> tuple[Any | None, str | None]:
        """Find specified pod or default to newest. Returns (pod, error_message)."""
        if not pod_name:
            return sorted_pods[0], None
        for p in sorted_pods:
            if p.metadata.name == pod_name:
                return p, None
        available_pods = [p.metadata.name for p in sorted_pods]
        return None, f"Pod '{pod_name}' not found. Available pods: {available_pods}"

    @staticmethod
    def _check_pod_pending(target_pod: Any) -> str | None:
        """Return status message if pod is still in Pending state, else None."""
        if target_pod.status.phase != "Pending":
            return None

        waiting_reasons = []
        statuses = (target_pod.status.init_container_statuses or []) + (
            target_pod.status.container_statuses or []
        )
        for cs in statuses:
            if cs.state and cs.state.waiting:
                waiting_reasons.append(
                    f"{cs.name}: {cs.state.waiting.reason or 'ContainerCreating'}"
                )
        reasons_str = f" ({', '.join(waiting_reasons)})" if waiting_reasons else ""
        return (
            f"Pod '{target_pod.metadata.name}' is still in 'Pending' state{reasons_str}. "
            "Logs are not available yet."
        )

    @staticmethod
    def _resolve_container(target_pod: Any, container: str | None) -> tuple[str | None, bool]:
        """Auto-detect container if not specified. Returns (container_name, is_init_container)."""
        if container:
            return container, False

        init_statuses = target_pod.status.init_container_statuses or []
        for init_status in init_statuses:
            if init_status.name != "storage-initializer" or not init_status.state:
                continue
            state = init_status.state
            if state.terminated and state.terminated.exit_code != 0:
                return "storage-initializer", True
            if state.waiting and state.waiting.reason in [
                "CrashLoopBackOff",
                "ImagePullBackOff",
                "ErrImagePull",
            ]:
                return "storage-initializer", True

        container_names = [c.name for c in (target_pod.spec.containers or [])]
        if "kserve-container" in container_names:
            return "kserve-container", False
        if container_names:
            return container_names[0], False
        return None, False

    @staticmethod
    def _build_log_header(
        selected_pod_name: str,
        selected_container: str | None,
        sorted_pods: list[Any],
        is_init_container: bool,
    ) -> str:
        """Build metadata header displayed above logs."""
        other_pods = [p.metadata.name for p in sorted_pods if p.metadata.name != selected_pod_name]
        header = [f"=== Pod: {selected_pod_name} | Container: {selected_container} ==="]
        if other_pods:
            suffix = f" and {len(other_pods) - 3} more)" if len(other_pods) > 3 else ")"
            header.append(f"(Other running replicas: {', '.join(other_pods[:3])}{suffix}")
        if is_init_container:
            header.append(
                "WARNING: Detected failure in 'storage-initializer' init container "
                "(check model storage path/credentials)."
            )
        return "\n".join(header)

    def get_predictor_logs(
        self,
        name: str,
        namespace: str = "default",
        pod_name: str | None = None,
        container: str | None = None,
        tail_lines: int = 100,
    ) -> str:
        """Fetch logs from a predictor pod of an InferenceService.

        Intelligently selects the newest pod and targets 'kserve-container'
        or 'storage-initializer' if a download failure occurred.
        """
        label_selector = f"serving.kserve.io/inferenceservice={name}"

        # 1. Safely query pods (handle RBAC / Namespace not found)
        try:
            pod_list = self.core_api.list_namespaced_pod(
                namespace=namespace, label_selector=label_selector
            )
        except ApiException as e:
            return f"Failed to list pods in namespace '{namespace}': {e.reason} ({e.status})"

        if not pod_list.items:
            return (
                f"No predictor pods found for InferenceService '{name}' in namespace '{namespace}'. "
                "Note: If scale-to-zero is enabled, the service may be idle. Send traffic to wake it up."
            )

        # 2. Sort pods by creation_timestamp descending (newest first)
        sorted_pods = sorted(
            pod_list.items,
            key=lambda p: p.metadata.creation_timestamp or 0,
            reverse=True,
        )

        # 3. Target specific pod or default to newest
        target_pod, err = self._find_target_pod(sorted_pods, pod_name)
        if err or not target_pod:
            return err or "No target pod found"

        selected_pod_name = target_pod.metadata.name

        # 4. Check if pod is still pending / container creating
        pending_msg = self._check_pod_pending(target_pod)
        if pending_msg:
            return pending_msg

        # 5. Resolve container: auto-detect if not explicitly provided
        selected_container, is_init_container = self._resolve_container(target_pod, container)

        # 6. Fetch the logs
        try:
            logs = self.core_api.read_namespaced_pod_log(
                name=selected_pod_name,
                namespace=namespace,
                container=selected_container,
                tail_lines=tail_lines,
            )
        except ApiException as e:
            return (
                f"Failed to fetch logs for pod '{selected_pod_name}', "
                f"container '{selected_container}': {e.reason} ({e.status})"
            )

        # 7. Build rich metadata header
        header_text = self._build_log_header(
            selected_pod_name, selected_container, sorted_pods, is_init_container
        )
        return f"{header_text}\n\n{logs or '<Empty log output>'}"
