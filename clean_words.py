# import random

# # Clean word list: remove duplicates, long words, and shuffle
# input_file = "words.txt"
# output_file = "words_clean.txt"

# unique_words = set()

# with open(input_file, "r", encoding="utf-8") as f:
#     for line in f:
#         for word in line.strip().split():
#             word = word.strip()
#             if len(word) < 8:  # only keep words <= 8 characters
#                 unique_words.add(word.lower())

# # Sort alphabetically first
# sorted_words = sorted(unique_words)

# # Then randomly scramble
# random.shuffle(sorted_words)

# # Write to file
# with open(output_file, "w", encoding="utf-8") as f:
#     for word in sorted_words:
#         f.write(word + "\n")

# print(f"Cleaned and scrambled {len(sorted_words)} words written to {output_file}")


import os

directory = r"C:\Users\mggam\OneDrive\Documents\UIUC\MS\Sem 2\ECE 598DA\code\results"

for filename in os.listdir(directory):
    if "aw" in filename:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            print("Deleting:", file_path)
            os.remove(file_path)
