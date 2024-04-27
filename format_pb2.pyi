from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class State(_message.Message):
    __slots__ = ["taskLength", "taskMaxLatency", "localCPU", "localMIPSTerm", "edgeCPUTerm", "cloudCPUTerm", "numberOfPes", "fileSize", "outputSize", "containerSize", "maxLatency"]
    TASKLENGTH_FIELD_NUMBER: _ClassVar[int]
    TASKMAXLATENCY_FIELD_NUMBER: _ClassVar[int]
    LOCALCPU_FIELD_NUMBER: _ClassVar[int]
    LOCALMIPSTERM_FIELD_NUMBER: _ClassVar[int]
    EDGECPUTERM_FIELD_NUMBER: _ClassVar[int]
    CLOUDCPUTERM_FIELD_NUMBER: _ClassVar[int]
    NUMBEROFPES_FIELD_NUMBER: _ClassVar[int]
    FILESIZE_FIELD_NUMBER: _ClassVar[int]
    OUTPUTSIZE_FIELD_NUMBER: _ClassVar[int]
    CONTAINERSIZE_FIELD_NUMBER: _ClassVar[int]
    MAXLATENCY_FIELD_NUMBER: _ClassVar[int]
    taskLength: float
    taskMaxLatency: float
    localCPU: float
    localMIPSTerm: float
    edgeCPUTerm: float
    cloudCPUTerm: float
    numberOfPes: float
    fileSize: float
    outputSize: float
    containerSize: float
    maxLatency: float
    def __init__(self, taskLength: _Optional[float] = ..., taskMaxLatency: _Optional[float] = ..., localCPU: _Optional[float] = ..., localMIPSTerm: _Optional[float] = ..., edgeCPUTerm: _Optional[float] = ..., cloudCPUTerm: _Optional[float] = ..., numberOfPes: _Optional[float] = ..., fileSize: _Optional[float] = ..., outputSize: _Optional[float] = ..., containerSize: _Optional[float] = ..., maxLatency: _Optional[float] = ...) -> None: ...

class TrainModelRequest(_message.Message):
    __slots__ = ["new_state", "reward", "is_done"]
    NEW_STATE_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    IS_DONE_FIELD_NUMBER: _ClassVar[int]
    new_state: State
    reward: float
    is_done: bool
    def __init__(self, new_state: _Optional[_Union[State, _Mapping]] = ..., reward: _Optional[float] = ..., is_done: bool = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ["action"]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: int
    def __init__(self, action: _Optional[int] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...
