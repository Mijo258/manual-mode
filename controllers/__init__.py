"""Plug-and-play control feature extractors; model loading occurs on construction."""

import contracts
from .base import BaseController, ControllerOutput
from .face_controller import FaceController
from .hand_controller import HandController
from .eye_controller import EyeController


__all__ = ["BaseController", "ControllerOutput", "FaceController", "HandController",
           "EyeController", "create_controller"]


def create_controller(mode: contracts.ControlMode | str,
                      config: dict | None = None) -> BaseController:
    try:
        selected = mode if isinstance(mode, contracts.ControlMode) else contracts.ControlMode(str(mode).strip().upper())
    except ValueError as exc:
        raise ValueError(f"Unsupported controller mode {mode!r}; choose FACE, HAND or EYE.") from exc
    constructors = {contracts.ControlMode.FACE: FaceController,
                    contracts.ControlMode.HAND: HandController,
                    contracts.ControlMode.EYE: EyeController}
    if selected not in constructors:
        raise ValueError(f"Unsupported controller mode {selected.value!r}; choose FACE, HAND or EYE.")
    return constructors[selected](config=config)
