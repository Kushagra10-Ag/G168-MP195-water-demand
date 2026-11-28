
from pathlib import Path

import yaml


def load(name):
    return yaml.safe_load(
        Path("k8s", name).read_text(encoding="utf-8")
    )


def test_deployment_matches_the_client_sheet():
    dep = load("deployment.yaml")

    container = dep["spec"]["template"]["spec"]["containers"][0]

    assert dep["metadata"]["name"] == "mp195-api"
    assert dep["spec"]["replicas"] == 3
    assert container["image"] == "mp195-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000


def test_probes_use_health_and_liveness_waits_longer():
    container = load("deployment.yaml")["spec"]["template"]["spec"]["containers"][0]

    ready = container["readinessProbe"]
    live = container["livenessProbe"]

    assert ready["httpGet"]["path"] == "/health" == live["httpGet"]["path"]
    assert live["initialDelaySeconds"] > ready["initialDelaySeconds"]


def test_service_selector_matches_pod_labels():
    dep = load("deployment.yaml")
    svc = load("service.yaml")

    labels = dep["spec"]["template"]["metadata"]["labels"]

    assert svc["metadata"]["name"] == "mp195-svc"
    assert svc["spec"]["selector"].items() <= labels.items()
    assert svc["spec"]["ports"][0]["port"] == 80
    assert svc["spec"]["ports"][0]["targetPort"] == 8000
    assert dep["spec"]["selector"]["matchLabels"].items() <= labels.items()
