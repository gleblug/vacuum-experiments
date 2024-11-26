

class KeysightProxy:
    def __init__(self):
        pass

    @staticmethod
    def value(instr, query: str) -> str:
        return instr.query(f'MEAS:{query}? 1E-3')

    @staticmethod
    def setAutoRange(instr, query: str):
        instr.query(f'MEAS:{query}? AUTO')