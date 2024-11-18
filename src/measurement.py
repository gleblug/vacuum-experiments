from datetime import datetime
import math
import time

import pyvisa
from hardware.interfaces import Ammeter
from hardware.keysight_truevolt import KeysightTruevolt
import logging
import sys


def writeCurrent(amp: Ammeter, filename: str):
	value = amp.current()
	now_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
	with open(filename, 'a') as f:
		f.write(f'{now_time}\t{value}\n')
	logging.debug("measurement written")

def measure():
	filename = f"../data/keysight_data_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv"
	keysight = KeysightTruevolt("USB0")
	with open(filename, "w") as f:
		f.write('time\tcurrent[A]\n')
	logging.info("Start measuring")
	while True:
		try:
			writeCurrent(keysight, filename)

			time_until_next = (1e6 - datetime.now().microsecond) / 1e6 - keysight.measure_time
			if time_until_next < 0:
				time_until_next -= math.floor(time_until_next)
			time.sleep(time_until_next)
		except KeyboardInterrupt:
			logging.info("STOP")
			return
		except ValueError as e:
			logging.warning(f"Error while parsing value: {e}")
			continue
		except pyvisa.errors.VisaIOError as e:
			logging.error("Disconnected. Trying to reconnect...")
			if keysight.reconnect():
				logging.error("Connected!")
				continue
			else:
				logging.critical("Failed.")
				return
		except Exception as e:
			logging.critical(f"STOP {e}")
			return

def main():
	logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
	measure()

if __name__ == '__main__':
	main()
