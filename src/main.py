import logging
import sys
import configparser
import argparse

from hardware.interfaces import Status
from hardware.multimeter_scpi import MultimeterSCPI
from measurer import Measurer

def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='Path to config.ini file.')
    args = parser.parse_args()

    logging.info("Load config...")
    config = configparser.ConfigParser()
    config.read(args.config_path)

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

    t = config.getfloat('Measurer', 'Timeout')
    d = config['Measurer']['Directory']
    measurer = Measurer(directory=d, meters=meters, timeout=t)
    measurer.measure()

if __name__ == '__main__':
    main()
