import logging
import sys
import configparser

from hardware.interfaces import Status
from hardware.multimeter_scpi import MultimeterSCPI
from measurer import Measurer
from hardware.keysight_proxy import KeysightProxy
from hardware.rigol_proxy import RigolProxy

def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Load config...")
    config = configparser.ConfigParser()
    config.read('config.ini')
    t = config.getfloat('Measurer', 'timeout')

    logging.info("Initialize hardware...")
    hardware_sections = [(config[s], s.split('.')[1]) for s in config.sections() if s.split('.')[0] == 'Hardware']
    meters = []
    for hs, name in hardware_sections:
        if not hs.getboolean('Use'):
            continue
        port = hs['Port']
        status = Status[hs['Status']]
        meter = MultimeterSCPI(port, name, status)
        meters.append(meter)
        logging.info(f'{name} connected.')

    measurer = Measurer(meters=meters, timeout=t)
    measurer.measure()

if __name__ == '__main__':
    main()
