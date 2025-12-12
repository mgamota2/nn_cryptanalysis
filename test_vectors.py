# --- PRESENT test runner: run known 80-bit-key test vectors and print bits ---
import binascii
from present_cipher import *

def hexclean(s):
    """Remove spaces/newlines and ensure even length."""
    s = ''.join(s.split())
    if len(s) % 2:
        s = '0' + s
    return s

def hex_to_bytes(s):
    return binascii.unhexlify(hexclean(s))

def bytes_to_bitstr(b, width):
    return format(int.from_bytes(b, byteorder='big'), '0{}b'.format(width))

def bytes_to_hex(b):
    return binascii.hexlify(b).decode('ascii').upper()

# Standard PRESENT test vectors (plaintext, key, expected-cipher) in hex
test_vectors = [
    ("0000000000000000", "00000000000000000000", "5579C1387B228445"),
    ("0000000000000000", "FFFFFFFFFFFFFFFFFFFF", "E72C46C0F5945049"),
    ("FFFFFFFFFFFFFFFF", "00000000000000000000", "A112FFC72F68417B"),
    ("FFFFFFFFFFFFFFFF", "FFFFFFFFFFFFFFFFFFFF", "3333DCD3213210D2"),
]

def run_tests():
    for idx, (pt_hex, key_hex, expected_hex) in enumerate(test_vectors, 1):
        pt = hex_to_bytes(pt_hex)
        key = hex_to_bytes(key_hex)
        expected = hex_to_bytes(expected_hex)

        # Create cipher
        cipher = Present(key)  # uses default 32 rounds and default S-box

        # Print inputs as hex and bits
        print(f"\n=== Test #{idx} ===")
        print("Plaintext (hex):", bytes_to_hex(pt))
        print("Plaintext (bits):", bytes_to_bitstr(pt, 64))
        print("Key       (hex):", bytes_to_hex(key))
        print("Key       (bits):", bytes_to_bitstr(key, 80))
        print("Expected C(hex):", bytes_to_hex(expected))
        print("Expected C(bits):", bytes_to_bitstr(expected, 64))

        # Encrypt
        enc = cipher.encrypt(pt)
        print("Computed C(hex):", bytes_to_hex(enc))
        print("Computed C(bits):", bytes_to_bitstr(enc, 64))
        ok_enc = enc == expected
        print("Encryption matches expected?:", ok_enc)

        # Decrypt (sanity)
        dec = cipher.decrypt(enc)
        print("Decrypted P(hex):", bytes_to_hex(dec))
        print("Decrypted P(bits):", bytes_to_bitstr(dec, 64))
        ok_dec = dec == pt
        print("Decryption matches original?:", ok_dec)

if __name__ == "__main__":
    run_tests()
