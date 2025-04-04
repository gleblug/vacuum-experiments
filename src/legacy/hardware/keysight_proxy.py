from hardware.interfaces import SCPIInstr

class KeysightProxy:
    def __init__(self):
        pass

    @staticmethod
    def configure(instr: SCPIInstr, query: str):
        instr.write(f'CONF:{query}')

    @staticmethod
    def value(instr: SCPIInstr, query: str) -> str:
        return instr.query(f'MEAS:{query}? 1E-3,3E-9')

    @staticmethod
    def setAutoRange(instr: SCPIInstr, query: str):
        instr.query(f'MEAS:{query}? AUTO')