from datetime import datetime
import time
import logging
from os import path
from threading import Thread, Lock
from dataclasses import dataclass

from hardware.interfaces import Meter

def nowTime() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

@dataclass
class Measure:
    time: str
    value: float

class Measurer:

    directory: str
    meters: list[tuple[Meter, str]]
    timeout: float
    startTime: float

    def __init__(self, directory="../data", meters: list[Meter] = [], timeout=.5):
        self.directory = directory
        self.meters = []
        self.timeout = timeout
        for meter in meters:
            fname = path.join(directory, f'{meter.name}_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv')
            self.meters.append((meter, fname))
            with open(fname, 'w') as f:
                f.write(f'time\t{meter.units()}\n')
        logging.info(f'Measurer initialized successfully with {len(self.meters)} meters and timeout = {self.timeout}s')


    def measure(self) -> None:
        if len(self.meters) == 0:
            logging.error("There are no meters to measure!")
            return
        logging.info("Started measuring.")
        self.startTime = time.monotonic()
        try:
            while True:
                threads = []
                for meter, fname in self.meters:
                    th = Thread(target=self._processMeter, args=(meter, fname))
                    threads.append(th)
                    th.start()
                for th in threads:
                    th.join()
                time.sleep(self.timeout)

        except KeyboardInterrupt:
            logging.error("Stopped by the user.")
        except Exception as e:
            logging.critical(f"Unknown error: {e}")
        logging.info("Finished measuring.")

    def _processMeter(self, meter: Meter, fname: str):
        mtime = nowTime()
        value = meter.value()
        meas = Measure(mtime, value)
        self._saveMeasure(fname, meas)

    @staticmethod
    def _saveMeasure(fname: str, meas: Measure):
        with open(fname, 'a') as f:
            f.write(f'{meas.time}\t{meas.value}\n')
            logging.debug("measurement written")
