from datetime import datetime
import time
import logging
from os import path
from threading import Thread, Lock, Timer
from dataclasses import dataclass

from hardware.interfaces import Meter

@dataclass
class Measure:
    time: float
    value: float

class Measurer:

    directory: str
    meters: list[tuple[Meter, str]]
    lock: Lock
    startTime: float

    def __init__(self, directory="../data", meters: list[Meter] = []):
        self.directory = directory
        self.meters = []
        self.lock = Lock()
        for meter in meters:
            fname = path.join(directory, f'{meter.name}_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv')
            self.meters.append((meter, fname))
            with open(fname, 'w') as f:
                f.write(f'time,s\t{meter.units()}\n')
            logging.debug("measurement written")

    def measure(self) -> None:
        logging.info("Start measuring")
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

        except KeyboardInterrupt:
            logging.info("STOP")
        except Exception as e:
            logging.critical(f"Unknown error: {e}")

    def _processMeter(self, meter: Meter, fname: str):
        mtime = time.monotonic() - self.startTime
        value = meter.value()
        meas = Measure(mtime, value)
        self._saveMeasure(fname, meas)
        # logging.info(f'name: {meter.name}, start: {start}, end: {meas.time}, value: {meas.value}')
        time.sleep(meter.timeout)

    @staticmethod
    def _saveMeasure(fname: str, meas: Measure):
        with open(fname, 'a') as f:
            f.write(f'{meas.time:.3f}\t{meas.value}\n')
            logging.debug("measurement written")
