import serial

dev = serial.Serial("COM4", baudrate=4800, parity="E", timeout=1)
dev.write(b'0001\t')
resp = dev.read()
dev.close()

print(resp)
print("END")