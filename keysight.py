import math
import easy_scpi as scpi
import time
from datetime import datetime

class KeysightAmpermeter:
	def __init__(self, port: str):
		self.port = port
		self.filename = f"keysight_data_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv"
		self.measure_time = .41 # seconds to one measurement
		self._init() # init hardware

	def _init(self):
		self.amp = scpi.Instrument(self.port)
		self.amp.connect()
		self.amp.write('SYST:LAB "MEASURING"')
		self.amp.write('CONF:CURR:DC')
		
		self.min_range = float(self.amp.query('CURR:DC:RANGE? MIN'))
		self.max_range = float(self.amp.query('CURR:DC:RANGE? MAX'))
		self.cur_range = self.max_range

	def __del__(self):
		self.amp.write('SYST:LAB ""')
		self.amp.disconnect()

	def measurement(self):
		try:
			value = float(self.amp.query(f'MEAS:CURR:DC? {self.cur_range:E}'))
			if ((value > .8 * self.cur_range) and (not math.isclose(self.cur_range, self.max_range))) \
				or ((value < .1 * self.cur_range) and (not math.isclose(self.cur_range, self.min_range))):
				value = float(self.amp.query(f'MEAS:CURR:DC? AUTO'))
				self.cur_range = float(self.amp.query('CURR:DC:RANGE?'))

			now_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
			with open(self.filename, 'a') as f:
				f.write(f'{now_time}\t{value}\n')
			
		except ValueError:
			return
		

def main():
	amp = KeysightAmpermeter("USB0")
	try:
		while True:
			amp.measurement()

			time_until_next = (1e6 - datetime.now().microsecond) / 1e6 - amp.measure_time
			if time_until_next < 0:
				time_until_next -= math.floor(time_until_next)
			time.sleep(time_until_next)
	except:
		return

if __name__ == '__main__':
	main()
