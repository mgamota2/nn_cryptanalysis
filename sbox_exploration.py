import math
import random

def walsh_transform_sbox(sbox):
    """Compute the nonlinearity of an S-box."""
    n = int(math.log2(len(sbox))) # number of bits
    max_walsh = 0

    for a in range(1, 1 << n):  # skip a=0, go to 1111
        for b in range(1 << n): #include b=0, go to 1111
            total = 0
            for x in range(1 << n):
                ax = bin(a & x).count('1') % 2 # bitwise dot product
                bphi = bin(b & sbox[x]).count('1') % 2 # bitwise dot product
                total += (-1) ** (ax ^ bphi)
            max_walsh = max(max_walsh, abs(total))

    nl = (2**(n - 1)) - (max_walsh / 2) # 2**(n-1) - 1/2(max_walsh)
    return nl


def random_sbox(n):
    """Generate a random S-box of size 2^n."""
    sbox = list(range(1 << n))
    random.shuffle(sbox)
    return sbox


def explore_sboxes(n, target_levels):
    """Search random S-boxes until examples with specific nonlinearity are found."""
    found = {}
    count = 0

    while len(found) < len(target_levels):
        sbox = random_sbox(n)
        nl = walsh_transform_sbox(sbox)
        count += 1

        if nl in target_levels and nl not in found:
            found[nl] = sbox
            print(f"Found S-box with nonlinearity {nl} after {count} trials:")
            print(sbox)
            print()
        if (count %100==0):
            print(f"Count is: {count}, found: {found}")

    print("All target nonlinearities found!")
    return found


if __name__ == "__main__":
    results = explore_sboxes(n=4, target_levels=[0,2,4])
    