"""Shared interfaces from the parent repository's docs/plan.md."""
from typing import Annotated, Any, Literal, NotRequired, Protocol, TypedDict, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DeviceState(TypedDict):
    id: str
    kind: Literal["drone", "rover", "camera"]
    connected: bool
    flying: bool
    battery_pct: int
    height_cm: int
    yaw_deg: int
    simulated: NotRequired[bool]
    flight_enabled: NotRequired[bool]
    fault: NotRequired[str]
    airborne_possible: NotRequired[bool]
    landing_requested: NotRequired[bool]
    connection_state: NotRequired[str]
    video_ready: NotRequired[bool]
    forward_range: NotRequired[dict[str, Any]]
    scan_capable: NotRequired[bool]
    pose: NotRequired[dict[str, float]]
    pose_source: NotRequired[str]
    depth_source: NotRequired[str]
    pilot_capable: NotRequired[bool]
    ready_for_control: NotRequired[bool]
    control_enabled: NotRequired[bool]
    armed: NotRequired[bool]
    link_open: NotRequired[bool]
    telemetry_age_s: NotRequired[float | None]
    video_age_s: NotRequired[float | None]
    landing_unconfirmed: NotRequired[bool]


@runtime_checkable
class DeviceAdapter(Protocol):
    id: str
    kind: str

    def state(self) -> DeviceState: ...
    def latest_frame(self) -> tuple[np.ndarray, float] | None: ...
    async def takeoff(self) -> None: ...
    async def land(self) -> None: ...
    async def hover(self) -> None: ...
    async def move(self, forward_cm: int, left_cm: int, up_cm: int) -> None: ...
    async def rotate(self, cw_deg: int) -> None: ...
    def rc(self, left_right: int, fwd_back: int, up_down: int, yaw: int) -> None: ...
    async def emergency(self) -> None: ...


class Detection(TypedDict):
    track_id: int
    box: tuple[int, int, int, int]
    confidence: float


class ChatEvent(TypedDict):
    type: Literal["chat"]
    role: Literal["assistant", "user"]
    text: str


class DeviceStateEvent(TypedDict):
    type: Literal["device_state"]
    device: DeviceState


class ToolCallEvent(TypedDict):
    type: Literal["tool_call"]
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class ConfirmRequestEvent(TypedDict):
    type: Literal["confirm_request"]
    request_id: str
    device_id: str
    image_url: str
    reason: str


class PhotoEvent(TypedDict):
    type: Literal["photo"]
    photo_id: str
    url: str
    caption: str


class MissionStatusEvent(TypedDict):
    type: Literal["mission_status"]
    status: Literal["started", "completed", "cancelled", "failed"]


Event = ChatEvent | DeviceStateEvent | ToolCallEvent | ConfirmRequestEvent | PhotoEvent | MissionStatusEvent


class ClientEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserMessage(ClientEvent):
    type: Literal["user_message"]
    text: str = Field(min_length=1, max_length=8000)


class ConfirmResponse(ClientEvent):
    type: Literal["confirm_response"]
    request_id: str = Field(min_length=1)
    approved: bool = Field(strict=True)


class StopAll(ClientEvent):
    type: Literal["stop_all"]


IncomingEvent = Annotated[UserMessage | ConfirmResponse | StopAll, Field(discriminator="type")]
