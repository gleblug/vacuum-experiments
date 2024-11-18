from datetime import datetime
import math
import easy_scpi as scpi

from hardware.interfaces import MultiStatus

class KeysightTruevolt:

	instr: scpi.Instrument
	status: MultiStatus
	min_range: float
	max_range: float
	cur_range: float

	def __init__(self, port: str):
		self.measure_time = .41 # seconds to one measurement
		self.status = MultiStatus.unknown

		self.instr = scpi.Instrument(port)
		self.instr.connect()
		self.setMsg("MEASURING")
		
	def __del__(self):
		self.instr.write('SYST:LAB ""')
		self.instr.disconnect()

	def setMsg(self, msg: str):
		self.instr.write(f'SYST:LAB "{msg}"')

	def current(self) -> float:
		self._confCurrentDC()

		value = float(self.instr.query(f'MEAS:CURR:DC? {self.cur_range:E}'))
		if ((value > .8 * self.cur_range) and (not math.isclose(self.cur_range, self.max_range))) \
			or ((value < .1 * self.cur_range) and (not math.isclose(self.cur_range, self.min_range))):
			value = float(self.instr.query(f'MEAS:CURR:DC? AUTO'))
			self.cur_range = float(self.instr.query('CURR:DC:RANGE?'))
		return value

	def _confCurrentDC(self) -> None:
		if self.status == MultiStatus.currentDC:
			return
		self.instr.write('CONF:CURR:DC')
		self.status = MultiStatus.currentDC

		self.min_range = float(self.instr.query('CURR:DC:RANGE? MIN'))
		self.max_range = float(self.instr.query('CURR:DC:RANGE? MAX'))
		self.cur_range = self.max_range

