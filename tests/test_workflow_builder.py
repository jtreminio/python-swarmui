from __future__ import annotations

from swarmui_clone.schemas import GenerationRequest
from swarmui_clone.services.workflow_builder import WorkflowBuilder


def test_workflow_builder_aligns_sizes_and_sets_seed():
    builder = WorkflowBuilder()
    request = GenerationRequest(
        prompt="prompt",
        model="model.safetensors",
        width=1025,
        height=1001,
    )

    workflow = builder.build(request, seed=42, prompt="hello", negative_prompt="bad")

    assert workflow["4"]["inputs"]["seed"] == 42
    assert workflow["5"]["inputs"]["width"] == 1024
    assert workflow["5"]["inputs"]["height"] == 1000
    assert workflow["6"]["inputs"]["text"] == "hello"
    assert workflow["7"]["inputs"]["text"] == "bad"


def test_workflow_builder_allows_custom_workflow_passthrough():
    builder = WorkflowBuilder()
    custom = {"1": {"class_type": "Anything", "inputs": {}}}
    request = GenerationRequest(
        prompt="prompt",
        model="model.safetensors",
        custom_workflow=custom,
    )

    workflow = builder.build(request, seed=1, prompt="x", negative_prompt="y")

    assert workflow == custom
