from typing import Protocol
from enum import Enum, auto

class Status(Enum):
    DISCONNECTED = auto()
    UNKNOWN = auto()
    CURR_DC = auto()
    CURR_AC = auto()
    VOLT_DC = auto()
    VOLT_AC = auto()
    RES_2W = auto()
    RES_4W = auto()

class SCPIProxy(Protocol):
    @staticmethod
    def setAutoRange(instr, query: str):
        pass

    @staticmethod
    def value(instr, query: str) -> float:
        return float('nan')

class Meter(Protocol):

    name: str
    timeout: float

    def value(self) -> float:
        return float("nan")

    def units(self) -> str:
        return ''