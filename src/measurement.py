from datetime import datetime
import math
import time

import pyvisa
from hardware.interfaces import Ammeter, Voltmeter
from hardware.keysight_truevolt import KeysightTruevolt
import logging
import sys


def nowTime() -> str:
	return 	datetime.now().strftime("%d.%m.%Y %H:%M:%S")

def writeCurrent(amp: Ammeter, filename: str):
	value = amp.current()
	with open(filename, 'a') as f:
		f.write(f'{nowTime()}\t{value}A\n')
	logging.debug("measurement written")

def writeVoltage(volt: Voltmeter, filename: str):
	value = volt.voltage()
	with open(filename, 'a') as f:
		f.write(f'{nowTime()}\t{value}V\n')
	logging.debug("measurement written")


def measure():
	filename = f"../data/keysight_data_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv"
	keysight = KeysightTruevolt("USB0")
	logging.info("Start measuring")
	while True:
		try:
			writeCurrent(keysight, filename)
			# writeVoltage(keysight, filename)

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
