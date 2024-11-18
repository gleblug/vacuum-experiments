from datetime import datetime
import logging
import math
import time
import easy_scpi as scpi
import pyvisa

from hardware.interfaces import MultiStatus

class KeysightTruevolt:

	instr: scpi.Instrument
	port: str
	connected: bool
	status: MultiStatus
	min_range: float
	max_range: float
	cur_range: float

	def __init__(self, port: str):
		self.measure_time = .41 # seconds to one measurement
		self.port = port
		self.connected = False
		self.connect()

	def connect(self):
		self.status = MultiStatus.unknown
		self.instr = scpi.Instrument(self.port)
		self.instr.connect()
		self.connected = True
		self.setMsg("MEASURING")

	def reconnect(self) -> bool:
		self.instr.disconnect()
		self.connected = False
		attempt = 1
		while self.connected == False:
			try:
				time.sleep(2 ** (attempt - 1))
				logging.info(f"Attempt {attempt}...")
				attempt += 1
				self.connect()
			except RuntimeError:
				logging.info("...failed")
				continue
			except pyvisa.errors.VisaIOError:
				self.instr.disconnect()
				time.sleep(1)
				self.instr.connect()
		return True

	def __del__(self):
		if not self.connected:
			return
		self.setMsg("")

	def setMsg(self, msg: str):
		self.instr.write(f'SYST:LAB "{msg}"')

	def current(self) -> float:
		query = 'CURR:DC'
		if self.status != MultiStatus.currentDC:
			self._configure(query)
		return self._getValueUpdRanges(query)

	def _configure(self, query: str) -> None:
		self.instr.write(f'CONF:{query}')
		self.status = MultiStatus.currentDC

		self.min_range = float(self.instr.query(f'{query}:RANGE? MIN'))
		self.max_range = float(self.instr.query(f'{query}:RANGE? MAX'))
		self.cur_range = self.max_range

	def _getValueUpdRanges(self, query: str) -> float:
		value = float(self.instr.query(f'MEAS:{query}? {self.cur_range:E}'))
		if ((value > .8 * self.cur_range) and (not math.isclose(self.cur_range, self.max_range))) \
			or ((value < .1 * self.cur_range) and (not math.isclose(self.cur_range, self.min_range))):
			value = float(self.instr.query(f'MEAS:{query}? AUTO'))
			self.cur_range = float(self.instr.query(f'SENS:{query}:RANGE?'))
		return value