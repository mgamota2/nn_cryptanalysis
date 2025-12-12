import random
import csv
import binascii

row=0
bits=64
filename = f"./random_data.txt"

while(row<1041):
    bit=0
    current_random_bitstring=''
    while (bit<bits):
        current_random_bitstring+=str(random.randint(0,1))
        bit+=1
    print(current_random_bitstring)
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_random_bitstring])
    row+=1
        
