def encode(row):
    representation = 0
    if row["frameData_characterData_0_key_A"]:
        representation += 64
    if row["frameData_characterData_0_key_B"]:
        representation += 32
    if row["frameData_characterData_0_key_C"]:
        representation += 16
    if row["frameData_characterData_0_key_U"]:
        representation += 8
    if row["frameData_characterData_0_key_L"]:
        representation += 4
    if row["frameData_characterData_0_key_R"]:
        representation += 2
    if row["frameData_characterData_0_key_D"]:
        representation += 1
    return representation
def decode(encoding):
    key = []

    if encoding >= 64:
        key.append("A")
        encoding -= 64
    if encoding >= 32:
        key.append("B")
        encoding -= 32
    if encoding >= 16:
        key.append("C")
        encoding -= 16
    if encoding >= 8:
        key.append("U")
        encoding -= 8
    if encoding >= 4:
        key.append("L")
        encoding -= 4
    if encoding >= 2:
        key.append("R")
        encoding -= 2
    if encoding >= 1:
        key.append("D")
        encoding -= 1
    return key
