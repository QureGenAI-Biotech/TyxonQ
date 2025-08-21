from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PulseInstruction:
    """A single pulse instruction targeting a hardware channel.

    Fields:
        channel: Hardware channel identifier (e.g., "d0", "u1").
        start: Start time in sample units (integer ticks).
        duration: Duration in sample units (ticks).
        waveform: Real or complex amplitude samples. Concrete dtype/shape is
            backend-specific; a Python list is accepted here for simplicity.
        metadata: Arbitrary metadata describing the pulse (shape, amp, sigma).

    Note:
        The unit convention follows sample counts to remain backend-agnostic.
        Conversion to seconds uses the schedule's sampling_rate_hz.
    """

    channel: str
    start: int
    duration: int
    waveform: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PulseSchedule:
    """A collection of timed pulse instructions with a sampling rate.

    Fields:
        sampling_rate_hz: Sampling frequency in Hertz for time conversion.
        instructions: Ordered list of pulse instructions.
        globals: Optional global parameters for template expansion or backends.
    """

    sampling_rate_hz: float
    instructions: List[PulseInstruction] = field(default_factory=list)
    globals: Dict[str, Any] = field(default_factory=dict)

    def append(self, instr: PulseInstruction) -> None:
        """Append an instruction to the schedule."""

        self.instructions.append(instr)

    def end_time(self) -> int:
        """Return the schedule end time in sample units.

        Defined as max over `start + duration` across all instructions, or 0
        when the schedule is empty.
        """

        if not self.instructions:
            return 0
        return max(i.start + i.duration for i in self.instructions)

    def duration_seconds(self) -> float:
        """Return the schedule duration in seconds based on sampling_rate_hz."""

        return self.end_time() / float(self.sampling_rate_hz)


