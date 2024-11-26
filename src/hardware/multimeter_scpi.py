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
    connected: bool
    cur_range: str

    def __init__(self, port: str, name: str, proxy, timeout: float = .5):
        self.port = port
        self.name = name
        self.proxy = proxy
        self.timeout = timeout
        self.status = Status.UNKNOWN
        self.connected = False
        self.connect()

    def __del__(self):
        if not self.connected:
            return
        self.instr.close()

    def connect(self):
        rm = pyvisa.ResourceManager()
        self.instr = rm.open_resource(self.port)
        self.connected = True

    def reconnect(self) -> bool:
        self.instr.close()
        attempt = 1
        while not self.connected:
            try:
                time.sleep(2 ** (attempt - 1))
                logging.info(f"Attempt {attempt}...")
                attempt += 1
                self.connect()
            except RuntimeError:
                logging.info("...failed")
                continue
            except pyvisa.errors.VisaIOError:
                self.instr.close()
                time.sleep(1)
        return True

    def setMode(self, status: Status) -> None:
        if self.status == status:
            return
        self._configure(status)

    def value(self) -> float:
        value = 0
        try:
            value = self._getValueUpdRanges()
        except ValueError as e:
            logging.warning(f"Error while parsing value: {e}")
        except pyvisa.errors.VisaIOError as e:
            logging.error("Disconnected. Trying to reconnect...")
            if self.reconnect():
                logging.error("Connected!")
            else:
                logging.critical("Failed.")
                raise RuntimeError(f"Cannot reconnect: {e}")
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

    def _getValueUpdRanges(self) -> float:
        value = float(self.proxy.value(self.instr, self.query()))
        return value
        # if ((value > .95 * float(self.cur_range)) or (value < .08 * float(self.cur_range)) and  \
        #     self.cur_range != self.min_range and self.cur_range != self.max_range):
        #     value = float(self.instr.query(f'MEAS:{self.query()}? AUTO'))
        #     self.cur_range = self.instr.query(f'SENS:{self.query()}:RANGE?')