from hardware.interfaces import SCPIInstr

class RigolProxy:
    def __init__(self):
        pass

    @staticmethod
    def configure(instr: SCPIInstr, query: str):
        instr.write(f'CONF:{query}')
        instr.write(f':{query}:NPLC 1')

    @staticmethod
    def value(instr: SCPIInstr, query: str) -> str:
        return instr.query(f'MEAS:{query}?')

    @staticmethod
    def setAutoRange(instr: SCPIInstr, query: str):
        instr.query(f'{query}:AUTO?')