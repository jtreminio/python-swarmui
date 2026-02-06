from __future__ import annotations

import pytest

from swarmui_clone.schemas import GenerationRequest
from swarmui_clone.services.workflow_builder import WorkflowBuildError, WorkflowBuilder


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


def test_workflow_builder_builds_flux_graph_when_model_name_matches():
    builder = WorkflowBuilder()
    request = GenerationRequest(
        prompt="prompt",
        negative_prompt="bad",
        model="flux1-dev.safetensors",
        width=1025,
        height=1001,
        cfg_scale=4.0,
    )

    workflow = builder.build(
        request,
        seed=123,
        prompt="a cat",
        negative_prompt="low quality",
        available_models={
            "CLIP": ["clip_l.safetensors", "t5xxl_fp16.safetensors"],
            "VAE": ["ae.safetensors"],
        },
    )

    assert workflow["10"]["class_type"] == "DualCLIPLoader"
    assert workflow["10"]["inputs"]["clip_name1"] == "clip_l.safetensors"
    assert workflow["10"]["inputs"]["clip_name2"] == "t5xxl_fp16.safetensors"
    assert workflow["13"]["inputs"]["vae_name"] == "ae.safetensors"
    assert workflow["4"]["inputs"]["cfg"] == 1.0
    assert workflow["5"]["inputs"]["width"] == 1024
    assert workflow["5"]["inputs"]["height"] == 992
    assert workflow["11"]["class_type"] == "CLIPTextEncodeFlux"
    assert workflow["12"]["class_type"] == "CLIPTextEncodeFlux"


def test_workflow_builder_flux_uses_zeroed_negative_when_empty():
    builder = WorkflowBuilder()
    request = GenerationRequest(
        prompt="prompt",
        model="flux-dev.safetensors",
    )

    workflow = builder.build(
        request,
        seed=7,
        prompt="hello",
        negative_prompt="",
        available_models={
            "CLIP": ["clip_l.safetensors", "t5xxl_fp16.safetensors"],
            "VAE": ["ae.safetensors"],
        },
    )

    assert workflow["12"]["class_type"] == "ConditioningZeroOut"
    assert workflow["12"]["inputs"]["conditioning"] == ["11", 0]


def test_workflow_builder_flux_requires_clip_and_vae():
    builder = WorkflowBuilder()
    request = GenerationRequest(
        prompt="prompt",
        model="flux-schnell.safetensors",
    )

    with pytest.raises(WorkflowBuildError):
        builder.build(
            request,
            seed=1,
            prompt="x",
            negative_prompt="y",
            available_models={},
        )


def test_workflow_builder_uses_parent_profile_for_z_image():
    builder = WorkflowBuilder()
    request = GenerationRequest(
        prompt="prompt",
        model="z-image/demo-model.safetensors",
        width=1025,
        height=1001,
    )

    workflow = builder.build(
        request,
        seed=5,
        prompt="hello",
        negative_prompt="",
    )

    assert workflow["10"]["class_type"] == "CLIPLoader"
    assert workflow["10"]["inputs"]["clip_name"] == "qwen_3_4b.safetensors"
    assert workflow["10"]["inputs"]["type"] == "stable_diffusion"
    assert workflow["13"]["inputs"]["vae_name"] == "Flux/ae.safetensors"
    assert workflow["5"]["class_type"] == "EmptySD3LatentImage"
    assert workflow["5"]["inputs"]["width"] == 1024
    assert workflow["5"]["inputs"]["height"] == 992
    assert workflow["12"]["class_type"] == "ConditioningZeroOut"


def test_workflow_builder_uses_parent_profile_for_flux():
    builder = WorkflowBuilder()
    request = GenerationRequest(
        prompt="prompt",
        model="Flux/demo-model.safetensors",
        flux_clip_name1="custom-clip.safetensors",
    )

    workflow = builder.build(
        request,
        seed=7,
        prompt="hello",
        negative_prompt="bad",
    )

    assert workflow["10"]["class_type"] == "CLIPLoader"
    assert workflow["10"]["inputs"]["clip_name"] == "custom-clip.safetensors"
    assert workflow["10"]["inputs"]["type"] == "flux2"
    assert workflow["13"]["inputs"]["vae_name"] == "Flux/flux2-vae.safetensors"
    assert workflow["5"]["class_type"] == "EmptyFlux2LatentImage"
    assert workflow["12"]["class_type"] == "CLIPTextEncode"
