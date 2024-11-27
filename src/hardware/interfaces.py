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

class SCPIInstr(Protocol):
    def write(self, msg: str) -> None:
        return

    def read(self) -> str:
        return ''

    def query(self, msg: str) -> str:
        return ''

class SCPIProxy(Protocol):
    @staticmethod
    def configure(instr, query: str):
        pass

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