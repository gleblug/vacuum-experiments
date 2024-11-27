import logging
import sys

from hardware.interfaces import Status
from hardware.multimeter_scpi import MultimeterSCPI
from measurer import Measurer
from hardware.keysight_proxy import KeysightProxy
from hardware.rigol_proxy import RigolProxy

def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Initialize devices...")
    keysight_port = "USB0::0x2A8D::0x1301::MY60004355::INSTR"
    rigol_port = "USB0::0x1AB1::0x0C94::DM3O262200553::INSTR"
    keysight = MultimeterSCPI(keysight_port, "keysight", KeysightProxy(), timeout=0)
    keysight.setMode(status=Status.CURR_DC)
    rigol = MultimeterSCPI(rigol_port, "rigol", RigolProxy(), timeout=0)
    rigol.setMode(status=Status.VOLT_DC)

    measurer = Measurer(meters=[keysight, rigol])
    measurer.measure()

    logging.info("End of the program.")

if __name__ == '__main__':
    main()
