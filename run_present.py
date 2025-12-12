from present_cipher import Present
import Padding
import sys
import math
import os
import random
import csv

random.seed(42) 

k = f"{random.getrandbits(80):020x}"
print(f"Current key: {k}")

rounds_to_test = [1,2,3,4,5,6,7,8,9,10,16,32]

if (len(sys.argv)>1):
    text=str(sys.argv[1])

if (len(sys.argv)>2):
    k=str(sys.argv[2])

print('Key:\t'+k)
print('--------')
print()

key = bytes.fromhex(k)

def walsh_transform_sbox(sbox):
    """Compute the nonlinearity of an S-box."""
    n = int(math.log2(len(sbox)))
    max_walsh = 0

    for a in range(1, 1 << n):  # skip a=0
        for b in range(1 << n):
            total = 0
            for x in range(1 << n):
                # dot products <a,x> and <b,Φ(x)> over GF(2)
                ax = bin(a & x).count('1') % 2
                bphi = bin(b & sbox[x]).count('1') % 2
                total += (-1) ** (ax ^ bphi)
            max_walsh = max(max_walsh, abs(total))

    # Compute nonlinearity
    nl = (2**(n - 1)) - (max_walsh // 2)
    return nl


import random


def avalanche_effect(cipher, num_samples=100):
    total_effect_across_samples = 0
    
    for _ in range(num_samples):
        # Generate random 64-bit plaintext
        plaintext = random.getrandbits(64).to_bytes(8, byteorder='big')
        original_ciphertext = cipher.encrypt(plaintext)
        
        total_effect = 0
        
        # Flip each bit in the plaintext
        for bit_position in range(64):
            plaintext_int = int.from_bytes(plaintext, byteorder='big')
            modified_plaintext_int = plaintext_int ^ (1 << (63 - bit_position))
            modified_plaintext = modified_plaintext_int.to_bytes(8, byteorder='big')

            # Encrypt modified plaintext
            modified_ciphertext = cipher.encrypt(modified_plaintext)

            # Count differing bits
            original_int = int.from_bytes(original_ciphertext, byteorder='big')
            modified_int = int.from_bytes(modified_ciphertext, byteorder='big')
            diff = original_int ^ modified_int
            
            bits_changed = bin(diff).count('1')
            total_effect += bits_changed
        
        # Average avalanche for this sample
        avg_effect_this_sample = total_effect / 64
        total_effect_across_samples += avg_effect_this_sample
    
    # Average across all samples
    return int(total_effect_across_samples / num_samples)



def bytes_to_bitlist(b: bytes):
    """Convert bytes to a list of bits (big-endian)."""
    return [int(bit) for byte in b for bit in f"{byte:08b}"]


def generate_ciphertext_plaintext_pairs(rounds, sbox, pbox, pbox_inv, pbox_name, wordlist_file="words_clean.txt"):
    """
    Generate plaintext–ciphertext pairs (and bit-level representations)
    using the PRESENT cipher with custom S-box and P-box.
    """

    cipher = Present(key, rounds=rounds, sbox=sbox, pbox=pbox, pbox_inv=pbox_inv)

    # Calculate metrics
    nl = walsh_transform_sbox(sbox)
    aw = round(avalanche_effect(cipher))
    
    filename = f"./new_data/{rounds}_rounds_nl{nl}_aw{aw}_p{pbox_name}_bits.csv"
    
    # Initialize cipher

    dataset = []

    with open(wordlist_file, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f.readlines() if w.strip()]

    for word in words:
        padded = Padding.appendPadding(word, blocksize=8, mode="CMS")
        plaintext_bytes = padded.encode()
        ciphertext_bytes = cipher.encrypt(plaintext_bytes)

        # Convert to bit vectors (lists of 0/1)
        plaintext_bits = bytes_to_bitlist(plaintext_bytes)
        ciphertext_bits = bytes_to_bitlist(ciphertext_bytes)

        dataset.append({
            "word": word,
            "plaintext_hex": plaintext_bytes.hex(),
            "ciphertext_hex": ciphertext_bytes.hex(),
            "plaintext_bits": "".join(map(str, plaintext_bits)),
            "ciphertext_bits": "".join(map(str, ciphertext_bits))
        })

    # Save dataset to CSV
    os.makedirs("./data", exist_ok=True)
    # with open(filename, "w", newline="", encoding="utf-8") as csvfile:
    #     fieldnames = ["word", "plaintext_hex", "ciphertext_hex", "plaintext_bits", "ciphertext_bits"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(dataset)

    # print(f"✓ Saved {len(dataset)} pairs to {filename}")


def random_generate_ciphertext_plaintext_pairs(rounds, sbox, pbox, pbox_inv, pbox_name, wordlist_file="random_data.txt"):
    """
    Generate random plaintext–ciphertext pairs using custom S-box and P-box.
    """

    cipher = Present(key, rounds=rounds, sbox=sbox, pbox=pbox, pbox_inv=pbox_inv)

    nl = walsh_transform_sbox(sbox)
    aw = round(avalanche_effect(cipher))
    
    filename = f"./new_data/random_{rounds}_rounds_nl{nl}_aw{aw}_p{pbox_name}_bits.csv"

    dataset = []

    with open(wordlist_file, "r", encoding="utf-8") as f:
        file_data = [w.strip() for w in f if w.strip()]

    for bitstring in file_data:
        # Validate bitstring length
        if len(bitstring) != 64:
            print(f"⚠️ Skipping invalid line (not 64 bits): {bitstring}")
            continue

        # Convert binary string to bytes
        int_value = int(bitstring, 2)
        plaintext_bytes = int_value.to_bytes(8, byteorder="big")

        # Encrypt
        ciphertext_bytes = cipher.encrypt(plaintext_bytes)

        # Convert to bit vectors
        plaintext_bits = bytes_to_bitlist(plaintext_bytes)
        ciphertext_bits = bytes_to_bitlist(ciphertext_bytes)

        dataset.append({
            "word": bitstring,
            "plaintext_hex": plaintext_bytes.hex(),
            "ciphertext_hex": ciphertext_bytes.hex(),
            "plaintext_bits": "".join(map(str, plaintext_bits)),
            "ciphertext_bits": "".join(map(str, ciphertext_bits))
        })

    os.makedirs("./new_data", exist_ok=True)
    # with open(filename, "w", newline="", encoding="utf-8") as csvfile:
    #     fieldnames = ["word", "plaintext_hex", "ciphertext_hex", "plaintext_bits", "ciphertext_bits"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(dataset)

    print(f"✓ Saved {len(dataset)} pairs to {filename}")


# Example usage
if __name__ == "__main__":
    # Define S-boxes
    default_sbox =     [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
    perfectly_linear = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf]
    nonlinearity_two = [0x1, 0x6, 0xD, 0x0, 0x4, 0x2, 0xA, 0x5, 0x7, 0x8, 0x3, 0xC, 0xB, 0xF, 0x9, 0xE]

    # Define P-boxes
    pbox_default = [0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,
                    4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,
                    8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,
                    12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63]
    
    pbox_trivial = list(range(64))  # Identity permutation (no diffusion)
    
    # Weak diffusion P-box (keeps bits more localized)
    pbox_weak = [0,1,2,3,16,17,18,19,32,33,34,35,48,49,50,51,
                 4,5,6,7,20,21,22,23,36,37,38,39,52,53,54,55,
                 8,9,10,11,24,25,26,27,40,41,42,43,56,57,58,59,
                 12,13,14,15,28,29,30,31,44,45,46,47,60,61,62,63]
    
    # Compute inverse P-boxes
    pbox_default_inv = [pbox_default.index(x) for x in range(64)]
    pbox_trivial_inv = [pbox_trivial.index(x) for x in range(64)]
    pbox_weak_inv = [pbox_weak.index(x) for x in range(64)]

    # Define all S-box and P-box combinations
    sboxes = [
        ("Default S-box (NL=4)", default_sbox),
        ("Linear S-box (NL=0)", perfectly_linear),
        ("Low NL S-box (NL=2)", nonlinearity_two),
    ]
    
    pboxes = [
        ("Default P-box", pbox_default, pbox_default_inv),
        ("Trivial P-box", pbox_trivial, pbox_trivial_inv),
        ("Weak P-box", pbox_weak, pbox_weak_inv),
    ]

    # Test all 9 combinations
    for sbox_name, sbox in sboxes:
        for pbox_name, pbox, pbox_inv in pboxes:
            name = f"{sbox_name} + {pbox_name}"
            print("\n" + "="*80)
            print(f"{name}")
            print("="*80)
            
            nl = walsh_transform_sbox(sbox)
            print(f"S-box Nonlinearity: {nl}")
            
                    
            print("\nGenerating datasets...")
            print("-" * 80)
            
            for rounds in rounds_to_test:
                generate_ciphertext_plaintext_pairs(
                    rounds=rounds, 
                    sbox=sbox,
                    pbox=pbox,
                    pbox_inv=pbox_inv,
                    pbox_name=pbox_name.split()[0],
                    wordlist_file="words_clean.txt"
                )
                random_generate_ciphertext_plaintext_pairs(
                    rounds=rounds,
                    sbox=sbox,
                    pbox=pbox,
                    pbox_inv=pbox_inv,
                    pbox_name=pbox_name.split()[0],
                    wordlist_file="random_data.txt"
                )
            
            print()