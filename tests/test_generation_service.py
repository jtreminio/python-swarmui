from __future__ import annotations

from swarmui_clone.services.generation import GenerationService


def test_extract_execution_error_from_history_status_messages():
    history = {
        "prompt-1": {
            "status": {
                "status_str": "error",
                "messages": [
                    [
                        "execution_error",
                        {
                            "node_type": "CLIPTextEncode",
                            "exception_message": "clip input is invalid: None",
                        },
                    ]
                ],
            }
        }
    }

    error = GenerationService._extract_execution_error("prompt-1", history)
    assert error == "CLIPTextEncode: clip input is invalid: None"


def test_extract_execution_error_returns_none_for_non_error_status():
    history = {
        "prompt-2": {
            "status": {
                "status_str": "success",
                "messages": [["execution_success", {"prompt_id": "prompt-2"}]],
            }
        }
    }

    error = GenerationService._extract_execution_error("prompt-2", history)
    assert error is None
