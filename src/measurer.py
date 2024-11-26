from datetime import datetime
import logging
from os import path
from threading import Thread, Lock
from dataclasses import dataclass
import time

from hardware.interfaces import Meter

@dataclass
class Measure:
    time: str
    value: float

class Measurer:

    directory: str
    meters: list[tuple[Meter, str]]
    lock: Lock

    def __init__(self, directory="../data", meters: list[Meter] = []):
        self.directory = directory
        self.meters = []
        self.lock = Lock()
        for meter in meters:
            fname = path.join(directory, f'{meter.name}_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv')
            self.meters.append((meter, fname))
            with open(fname, 'w') as f:
                f.write(f'time\t{meter.units()}\n')
            logging.debug("measurement written")

    @staticmethod
    def nowTime() -> str:
        return  datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f")

    def measure(self) -> None:
        threads = []
        try:
            for meter, fname in self.meters:
                th = Thread(target=self._processMeter, args=(meter, fname))
                threads.append(th)
                th.start()
            while True:
                cmd = input()

        except KeyboardInterrupt:
            logging.info("STOP")
        except Exception as e:
            logging.critical(f"Unknown error: {e}")

        self.lock.acquire()
        for th in threads:
            th.join()

    def _processMeter(self, meter: Meter, fname: str):
        while True:
            if self.lock.locked():
                return
            value = meter.value()
            meas = Measure(self.nowTime(), value)
            self._saveMeasure(fname, meas)
            time.sleep(meter.timeout)

    @staticmethod
    def _saveMeasure(fname: str, meas: Measure):
        with open(fname, 'a') as f:
            f.write(f'{meas.time}\t{meas.value}\n')
            logging.debug("measurement written")
