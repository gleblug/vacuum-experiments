from typing import Protocol
import enum

class MultiStatus(enum.Enum):
	unknown = 0
	currentDC = 1
	currentAC = 2
	voltageDC = 3
	voltageAC = 4
	resistance2W = 5
	resistance4W = 6

class Ammeter(Protocol):
	def current(self) -> float:
		return float("nan")

class Voltmeter(Protocol):
	def voltage(self) -> float:
		return float("nan")

class Ohmmeter(Protocol):
	def resistance(self) -> float:
		return float("nan")