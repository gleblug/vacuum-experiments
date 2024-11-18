from datetime import datetime
import math
import easy_scpi as scpi

from hardware.interfaces import MultiStatus

class KeysightTruevolt:

	instr: scpi.Instrument
	port: str
	status: MultiStatus
	min_range: float
	max_range: float
	cur_range: float

	def __init__(self, port: str):
		self.measure_time = .41 # seconds to one measurement
		self.status = MultiStatus.unknown
		self.port = port
		self.connect()

	def connect(self):
		self.instr = scpi.Instrument(self.port)
		self.instr.connect()
		self.setMsg("MEASURING")

	def __del__(self):
		self.instr.write('SYST:LAB ""')
		self.instr.disconnect()

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