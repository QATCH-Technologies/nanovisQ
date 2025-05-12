import hashlib as h
import random as n


class GH_Security:

    ALPHANUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    GHAPIURL = ['myyux:', '', 'fun)lnymzg)htr', 'wjutx', 'VFYHM(Yjhmstqtlnjx',
                'sfstanxV', 'fhyntsx', 'fwynkfhyx']
    CIPHERTEXT = ['A', 'p', 'J', 'R', 'Y', 'T', 'J', 'A', 'A', 'L', 'A', 'Be',
                  'V', 'H', 'J', '3', 'y', 'X', 'bBzBLBG', 'p', '0', 'z', 'xBB',
                  'QBBBDBq', 'R', 'A', 'BFBsBb', 'h', 'w', '9BV', 'h', 'JBBBq',
                  'k', 'z', 'L', '6', 'G', 'j', 'o', '2', 'aBL', 'X', 'FBq',
                  'D', 'K', 'B', 'N', 'Q', 'v', 'SC', 'BRBB', 'f', 'G', 'zBc',
                  'v', 'y', 'H', 'pB8', 'q', 'b', 'XBf', 'qBE', '5', 'kB', 'A',
                  'x', '0BV', 'H', 's', 't', 'n', 'g', 'kBB', 'K', 'T', '0Br',
                  'b', '9Bh', 'p', 'J9327d1']

    def __init__(self) -> None:
        n.seed(42)  # the meaning of life, the universe, and everything

    def caesar_decipher(self, text: str, shift: int = 3) -> str:
        result = []
        if abs(shift) == 3:
            shift = self.random() if abs(shift) == shift else -self.random()
        for char in text:
            if char.isdigit():
                base = ord('0')
                result.append(chr((ord(char) - base + shift) % 10 + base))
            elif char.isupper():
                base = ord('A')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            elif char.islower():
                base = ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            elif ord(char) in range(32, 48):
                base = 32
                result.append(chr((ord(char) - base + shift) %
                              len(range(32, 48)) + base))
            elif ord(char) in range(58, 65):
                base = 58
                result.append(chr((ord(char) - base + shift) %
                              len(range(58, 65)) + base))
            elif ord(char) in range(91, 97):
                base = 91
                result.append(chr((ord(char) - base + shift) %
                              len(range(91, 97)) + base))
            elif ord(char) in range(123, 127):
                base = 123
                result.append(chr((ord(char) - base + shift) %
                              len(range(123, 127)) + base))
        return ''.join(result)

    def xor_decipher(self, data: bytes, key: str) -> str:
        return ''.join([chr(b ^ ord(key[i % len(key)])) for i, b in enumerate(data)])

    def decode_alphanum(self, encoded: str) -> bytes:
        base = len(self.ALPHANUM)
        data = []
        for i in range(0, len(encoded), 2):
            high = self.ALPHANUM.index(encoded[i])
            low = self.ALPHANUM.index(encoded[i+1])
            data.append(high * base + low)
        return bytes(data)

    def random(self) -> int:
        return n.randint(1, 25)

    def generate_checksum(self, text: str, salt: bytes, length: int = 6) -> str:
        salted = text.encode() + salt
        sha = h.sha256(salted).hexdigest()
        return sha[:length]

    def deobfuscate(self, obf: list[str], key: str, shift: int = 3, salt_len: int = 2, checksum_len: int = 6) -> str:
        salt_chars = salt_len * 2
        obf_str = "A".join([a if a != "A" else "" for a in obf])
        encoded_data = obf_str[:-salt_chars - checksum_len]
        salt_encoded = obf_str[-salt_chars - checksum_len:-checksum_len]
        checksum = obf_str[-checksum_len:]
        xor_bytes = self.decode_alphanum(encoded_data)
        salt = self.decode_alphanum(salt_encoded)
        caesar = self.xor_decipher(xor_bytes, key)
        original = self.caesar_decipher(caesar, shift)
        expected_checksum = self.generate_checksum(
            original, salt, checksum_len)
        if expected_checksum != checksum:
            raise ValueError(
                "Checksum mismatch! Data may have been tampered with.")
        if not original.startswith("Bearer"):
            raise ValueError(
                "Deobfuscate error! Data may have been tampered with.")
        return original

    def authorize(self, header: dict, key: str) -> dict:
        try:
            secure_header = header.copy()
            secure_header["Authorization"] = self.deobfuscate(
                self.CIPHERTEXT, key)
            secure_header = dict(sorted(secure_header.items()))
        except:
            pass
        return secure_header
