import logging
import sys

from hardware.interfaces import Status
from hardware.multimeter_scpi import MultimeterSCPI
from measurer import Measurer
from hardware.keysight_proxy import KeysightProxy
from hardware.rigol_proxy import RigolProxy

def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    keysight_port = "USB0::0x2A8D::0x1301::MY60004355::INSTR"
    rigol_port = "USB0::0x1AB1::0x0C94::DM3O262200553::INSTR"
    keysight = MultimeterSCPI(keysight_port, "keysight", KeysightProxy())
    keysight.setMode(status=Status.CURR_DC)
    rigol = MultimeterSCPI(rigol_port, "rigol", RigolProxy())
    rigol.setMode(status=Status.VOLT_DC)

    logging.info("Start measuring")

    measurer = Measurer(meters=[rigol, keysight])
    measurer.measure()

if __name__ == '__main__':
    main()
