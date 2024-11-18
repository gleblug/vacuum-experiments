from datetime import datetime
import math
import time
from hardware.interfaces import Ammeter
from hardware.keysight_truevolt import KeysightTruevolt

def measureCurrent(amp: Ammeter, filename: str):
	value = amp.current()
	now_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
	with open(filename, 'a') as f:
		f.write(f'{now_time}\t{value}\n')
		
def main():
	filename = f"keysight_data_{datetime.now().strftime("%Y-%m-%d_%H-%M-%s")}.csv"
	keysight = KeysightTruevolt("USB0")
	while True:
		try:
			measureCurrent(keysight, filename)

			time_until_next = (1e6 - datetime.now().microsecond) / 1e6 - keysight.measure_time
			if time_until_next < 0:
				time_until_next -= math.floor(time_until_next)
			time.sleep(time_until_next)
		except ValueError:
			continue

if __name__ == '__main__':
	main()
