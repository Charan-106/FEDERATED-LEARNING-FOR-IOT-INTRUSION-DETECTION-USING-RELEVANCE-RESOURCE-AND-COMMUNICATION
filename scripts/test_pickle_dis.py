import pickletools

with open("P:/FLmodel/label_encoder.pkl", "rb") as f:
    data = f.read()

print("Disassembling label_encoder.pkl:")
try:
    pickletools.dis(data)
except Exception as e:
    print("Disassembly error:", e)
