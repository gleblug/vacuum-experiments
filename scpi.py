import math
import time
import easy_scpi as scpi

msg = "REMOTE MEAS"

amp = scpi.Instrument("USB0")
amp.connect()
amp.write(f'SYST:LAB "{msg}"')
amp.write('CONF:CURR:DC')

min_range = float(amp.query('CURR:DC:RANGE? MIN'))
max_range = float(amp.query('CURR:DC:RANGE? MAX'))
cur_range = max_range

try:
	while True:
		value = float(amp.query(f'MEAS:CURR:DC? {cur_range:E}'))
		if (value > .8 * cur_range and cur_range < max_range) or (value < .1 * cur_range and cur_range > min_range):
			value = amp.query(f'MEAS:CURR:DC? AUTO')
			cur_range = float(amp.query('CURR:DC:RANGE?'))
		print(value)
		time.sleep(1)

except KeyboardInterrupt:
	amp.write('SYST:LAB ""')
	amp.disconnect()

def measurement_to_file():
	try:
		value = float(amp.query(f'MEAS:CURR:DC? {cur_range:E}'))
		if (value > .8 * cur_range and cur_range < max_range) or (value < .1 * cur_range and cur_range > min_range):
			value = amp.query(f'MEAS:CURR:DC? AUTO')
			cur_range = float(amp.query('CURR:DC:RANGE?'))
		print(value)
		time.sleep(1)
	except NameError:
		return
	except:
		return