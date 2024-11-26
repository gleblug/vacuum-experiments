

class RigolProxy:
    def __init__(self):
        pass

    @staticmethod
    def value(instr, query: str) -> str:
        return instr.query(f'MEAS:{query}?')

    @staticmethod
    def setAutoRange(instr, query: str):
        instr.query(f'{query}:AUTO?')