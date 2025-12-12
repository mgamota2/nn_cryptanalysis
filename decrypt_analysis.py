import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from present_cipher import Present
from Padding import removePadding
import random
import matplotlib.pyplot as plt
import re

random_bits = True

clean = True
# rounds = [1,2,3,4,5,6,7,8,9,10,16,32]
# rounds = [1,2,3,4,5,6,7,8]
rounds=[8]


# Replace with your key
random.seed(42) 
k = f"{random.getrandbits(80):020x}"
print(f"Current key: {k}")
key = bytes.fromhex(k)


# Example S-boxes
default_sbox =     [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
perfectly_linear = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf]
# nonlinearity_two = [0x4, 0xa, 0x1, 0x5, 0x9, 0xb, 0xd, 0xf, 0x3, 0x6, 0x8, 0x2, 0xe, 0x0, 0x7, 0xc]
nonlinearity_two = [0x1, 0x6, 0xD, 0x0, 0x4, 0x2, 0xA, 0x5, 0x7, 0x8, 0x3, 0xC, 0xB, 0xF, 0x9, 0xE]


sboxes = [perfectly_linear, nonlinearity_two, default_sbox]
sbox_names = ["perfectly_linear", "nonlinearity_two", "default_sbox"]

def bits_to_bytes(bitstring):
    return int(bitstring, 2).to_bytes(8, "big")

def bitstring_to_vector(bitstring):
    return np.array([int(b) for b in bitstring], dtype=float)

results_dir = "./results"

if clean:
    for filename in os.listdir(results_dir):
        if "decrypted" in filename:
            file_path = os.path.join(results_dir, filename)
            print("Removing:", file_path)
            os.remove(file_path)

# summary storage for plotting
summary = {}  # {rounds: list of dicts {"nonlinearity":..., "avg_cos":..., "bit_acc":...}}

for filename in os.listdir(results_dir):
    if not filename.endswith(".csv"):
        continue
    if "decryption_summary" in filename:
        print("Skipping summary csv")
        continue
    if "decrypted" in filename:
        print("Already decrypted: ", filename)
        continue

    print(f"\n=== Analyzing {filename} ===")

    # Remove extension and split into parts
    parts = filename.replace(".csv", "").split("_")

    # Handle random_ prefix if present
    if parts[0] == "random":
        parts = parts[1:]  # remove "random"

    # Expect something like ["r8", "nl4", "hs1024"]
    curr_rounds = int(parts[0][1:])  # e.g. "r8" -> 8
    curr_nl = int(parts[1][2:])      # e.g. "nl4" -> 4
    curr_sbox = sboxes[curr_nl // 2]
    curr_sbox_name = sbox_names[curr_nl // 2]

    cipher = Present(key, rounds=curr_rounds, sbox=curr_sbox)

    path = os.path.join(results_dir, filename)
    print(path)

    # --- Read bit-level accuracy from row 2 ---
    with open(path, "r") as f:
        line = f.readlines()[1]  # row 2
        if "Bit-Level Accuracy" in line:
            bit_acc = float(line.strip().split(":")[1].replace("%",""))
        else:
            bit_acc = None

    import ast  # safer than json.loads for inline lists

    # header is on the 3rd line
    df = pd.read_csv(path, header=2)

    predicted_msgs = []
    actual_msgs = []
    cosines = []

    for _, row in df.iterrows():
        # convert string of probs → list of floats
        pred_probs = np.array(ast.literal_eval(row["prediction"]), dtype=float)
        actual_bits = np.array(list(map(int, row["actual_bits"])), dtype=float)

        # ensure equal length
        n = min(len(pred_probs), len(actual_bits))
        v_pred, v_actual = pred_probs[:n], actual_bits[:n]

        # cosine similarity (NumPy)
        cos = np.dot(v_pred, v_actual) / (np.linalg.norm(v_pred) * np.linalg.norm(v_actual))
        cosines.append(float(cos))

        # bitstring from probabilities
        pred_bits = ''.join('1' if p >= 0.5 else '0' for p in v_pred)
        pred_bytes = bits_to_bytes(pred_bits)
        actual_bytes = bits_to_bytes(row["actual_bits"])

        decrypted_pred = cipher.decrypt(pred_bytes).decode(errors="ignore")
        decrypted_actual = cipher.decrypt(actual_bytes).decode(errors="ignore")

        # remove everything except English letters
        if "random" not in filename:
            decrypted_pred = re.sub(r'[^A-Za-z]', '', decrypted_pred)
            decrypted_actual = re.sub(r'[^A-Za-z]', '', decrypted_actual)

        predicted_msgs.append(decrypted_pred)
        actual_msgs.append(decrypted_actual)

    for predicted, actual in (zip(predicted_msgs, actual_msgs)):
        if predicted != actual and curr_nl<2 and curr_rounds<5:
            print(f"\n Discrepancy with curr_rounds: {curr_rounds} and curr_nl: {curr_nl}")
            print(f"Predicted: {predicted} \t Actual: {actual}")

    # add results directly to the same DataFrame
    df["decrypted_predicted"] = predicted_msgs
    df["decrypted_actual"] = actual_msgs
    df["cosine_similarity"] = cosines

    out_path = os.path.join(results_dir, f"decrypted_{filename}")
    df.to_csv(out_path, index=False)
    print(f"Saved decrypted analysis to {out_path}")
    avg_cos = np.mean(cosines)
    print(f"Average cosine similarity: {avg_cos}")

    # store in summary
    summary.setdefault(curr_rounds, []).append({
        "nonlinearity": curr_sbox_name,
        "avg_cos": avg_cos,
        "bit_acc": bit_acc
    })


def plot_rounds(round_list, summary, fig_title, save_path):
    # map nonlinearity names to numeric x values
    nonlin_x = {"perfectly_linear": 0, "nonlinearity_two": 2, "default_sbox": 4}

    n_rounds = len(round_list)
    fig, axes = plt.subplots(n_rounds, 2, figsize=(12, 4*n_rounds), squeeze=False)
    fig.suptitle(fig_title, fontsize=16, y=1.02)

    for i, rounds in enumerate(round_list):
        if rounds not in summary:
            continue
        data = summary[rounds]
        # sort by x value
        data_sorted = sorted(data, key=lambda x: nonlin_x[x["nonlinearity"]])
        x_vals = [nonlin_x[d["nonlinearity"]] for d in data_sorted]
        avg_cos_vals = [d["avg_cos"] for d in data_sorted]
        bit_acc_vals = [d["bit_acc"] for d in data_sorted]

        # --- Cosine similarity ---
        axes[i, 0].bar(x_vals, avg_cos_vals, color='skyblue', width=1.0)
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].set_xticks(x_vals)
        axes[i, 0].set_xlabel("Nonlinearity")
        axes[i, 0].set_ylabel("Cosine Similarity")
        axes[i, 0].set_title(f"Rounds = {rounds}")
        for x, y in zip(x_vals, avg_cos_vals):
            y_pos = 0.90  # clamp so label doesn't go above axis
            axes[i, 0].text(x, y_pos, f"{y:.3f}", ha='center', va='bottom')

        # --- Bit-level accuracy ---
        axes[i, 1].bar(x_vals, bit_acc_vals, color='salmon', width=1.0)
        axes[i, 1].set_ylim(0, 100)
        axes[i, 1].set_xticks(x_vals)
        axes[i, 1].set_xlabel("Nonlinearity")
        axes[i, 1].set_ylabel("Bit-Level Accuracy (%)")
        axes[i, 1].set_title(f"Rounds = {rounds}")
        for x, y in zip(x_vals, bit_acc_vals):
            y_pos = 90  # clamp so label doesn't go above axis
            axes[i, 1].text(x, y_pos, f"{y:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# # Save first figure: rounds 2,4,6
# plot_rounds([2,4,6], summary, "Performance vs Nonlinearity (Rounds 2,4,6)", "./results/report_plot246.png")

# # Save second figure: rounds 10,16,32
# plot_rounds([10,16,32], summary, "Performance vs Nonlinearity (Rounds 10,16,32)", "./results/report_plot101632.png")

for i in range(0, len(rounds), 3):  # group 3 rounds per figure
    group = rounds[i:i+3]
    plot_rounds(group, summary,
                f"Performance vs Nonlinearity (Rounds {group})",
                f"./results/report_rounds_{'_'.join(map(str, group))}.png")