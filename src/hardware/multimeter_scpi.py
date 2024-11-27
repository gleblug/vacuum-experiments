import logging
import math
import pyvisa

from hardware.interfaces import SCPIProxy, Status
from hardware.keysight_proxy import KeysightProxy
from hardware.rigol_proxy import RigolProxy

class MultimeterSCPI:

    port: str
    name: str
    status: Status
    proxy: SCPIProxy

    def __init__(self, port: str, name: str, status: Status):
        self.port = port
        self.name = name
        self.status = Status.DISCONNECTED
        self._connect()
        self.proxy = self._chooseProxy()
        self._configure(status)
        logging.debug(f"{self.name} successfully initialized on port {self.port}!")

    def __del__(self):
        self._disconnect()

    def _chooseProxy(self) -> SCPIProxy:
        idn = self.instr.query("*IDN?")
        idn = idn.lower()
        if 'keysight' in idn:
            return KeysightProxy()
        if 'rigol' in idn:
            return RigolProxy()
        raise RuntimeError('Unavailable hardware!')

    def _connect(self):
        rm = pyvisa.ResourceManager()
        self.instr = rm.open_resource(self.port)
        self.status = Status.UNKNOWN

    def _disconnect(self):
        self.instr.clear()
        self.instr.close()
        self.status = Status.DISCONNECTED

    def connected(self):
        return self.status != Status.DISCONNECTED

    def value(self) -> float:
        value = .0
        try:
            value = float(self.proxy.value(self.instr, self.query()))
        except ValueError as e:
            logging.warning(f"Error while parsing value: {e}")
        except pyvisa.errors.VisaIOError as e:
            logging.error("Disconnected.")
            return math.nan
        return value

    def units(self) -> str:
        return self.status.name

    def query(self) -> str:
        match self.status:
            case Status.CURR_DC:
                return 'CURR:DC'
            case Status.CURR_AC:
                return 'CURR:AC'
            case Status.VOLT_DC:
                return 'VOLT:DC'
            case Status.VOLT_AC:
                return 'VOLT:AC'
            case Status.UNKNOWN | _:
                return ''

    def _configure(self, status: Status) -> None:
        self.status = status
        self.proxy.configure(self.instr, self.query())
        # self.proxy.setAutoRange(self.instr, self.query())
        # self.cur_range = self.max_range