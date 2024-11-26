import logging
import math
import time
import pyvisa

from hardware.interfaces import SCPIProxy, Status

class MultimeterSCPI:

    port: str
    name: str
    proxy: SCPIProxy
    timeout: float
    status: Status

    def __init__(self, port: str, name: str, proxy, timeout: float = .5):
        self.port = port
        self.name = name
        self.proxy = proxy
        self.timeout = timeout
        self.connect()
        logging.info(f"{name} successfully initialized!")

    def __del__(self):
        self.disconnect()

    def connect(self):
        rm = pyvisa.ResourceManager()
        self.instr = rm.open_resource(self.port)
        self.status = Status.UNKNOWN

    def disconnect(self):
        self.instr.clear()
        self.instr.close()
        self.status = Status.DISCONNECTED

    def connected(self):
        return self.status != Status.DISCONNECTED

    def setMode(self, status: Status) -> None:
        if self.status == status:
            return
        self._configure(status)

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
        self.instr.write(f'CONF:{self.query()}')
        # self.proxy.setAutoRange(self.instr, self.query())
        # self.cur_range = self.max_range