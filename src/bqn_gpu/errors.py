"""Typed failures at the BQN/backend semantic boundary."""


class BQNGPUError(Exception):
    """Base class for expected bqn-gpu failures."""


class DomainError(BQNGPUError):
    """The supplied value is outside the backend's supported domain."""


class ShapeError(BQNGPUError):
    """Argument shapes do not satisfy BQN agreement rules."""


class DeviceError(BQNGPUError):
    """The requested execution device is unavailable or inconsistent."""


class UnsupportedPrimitive(BQNGPUError):
    """The requested BQN primitive or valence is not implemented."""


class SourceError(BQNGPUError):
    """BQN source is invalid or outside the source frontend's current subset."""
