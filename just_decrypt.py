"""
decrypt_results.py - Decrypt cipher predictions and compute metrics
"""
import os
import pandas as pd
import numpy as np
from present_cipher import Present
import random
import re
import ast
from typing import Dict, List, Tuple

# Configuration
RANDOM_SEED = 42
RESULTS_DIR = "./new_results"
CLEAN_EXISTING = True

# S-box definitions
SBOXES = {
    "perfectly_linear": [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf],
    "nonlinearity_two": [0x1, 0x6, 0xD, 0x0, 0x4, 0x2, 0xA, 0x5, 0x7, 0x8, 0x3, 0xC, 0xB, 0xF, 0x9, 0xE],
    "default_sbox": [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
}

NONLINEARITY_MAP = {
    0: "perfectly_linear",
    2: "nonlinearity_two", 
    4: "default_sbox"
}

# Define P-boxes
pbox_default = [0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,
                4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,
                8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,
                12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63]

pbox_trivial = list(range(64))  # Identity permutation (no diffusion)

pbox_weak = [0,1,2,3,16,17,18,19,32,33,34,35,48,49,50,51,
             4,5,6,7,20,21,22,23,36,37,38,39,52,53,54,55,
             8,9,10,11,24,25,26,27,40,41,42,43,56,57,58,59,
             12,13,14,15,28,29,30,31,44,45,46,47,60,61,62,63]

# Compute inverse P-boxes
pbox_default_inv = [pbox_default.index(x) for x in range(64)]
pbox_trivial_inv = [pbox_trivial.index(x) for x in range(64)]
pbox_weak_inv = [pbox_weak.index(x) for x in range(64)]

pboxes = [
    ("Default P-box", pbox_default, pbox_default_inv),
    ("Trivial P-box", pbox_trivial, pbox_trivial_inv),
    ("Weak P-box", pbox_weak, pbox_weak_inv),
]


def bits_to_bytes(bitstring: str) -> bytes:
    """Convert bitstring to bytes."""
    return int(bitstring, 2).to_bytes(8, "big")


def parse_filename(filename: str) -> Dict:
    """Parse filename to extract all parameters.
    
    Args:
        filename: CSV filename (e.g., "random_r8_nl4_hs1024.csv" or "r10_nl4_aw2.17_hs256.csv")
        
    Returns:
        Dictionary with parsed parameters including:
        - rounds: number of cipher rounds
        - nl: nonlinearity level (0, 2, or 4)
        - sbox_name: name of S-box used
        - hidden_size: neural network hidden layer size
        - avalanche_effect: avalanche effect parameter (if present)
        - is_random: whether plaintext is random bits
        - plaintext_type: "random" or "text"
    """
    parts = filename.replace(".csv", "").split("_")
    
    # Check for random vs text-based
    is_random = parts[0] == "random"
    if is_random:
        parts = parts[1:]
    
    rounds = int(parts[0][1:])  # "r8" -> 8
    nl = int(parts[1][2:])      # "nl4" -> 4
    
    # Extract hidden size if present
    hidden_size = None
    for part in parts:
        if part.startswith("hs"):
            hidden_size = int(part[2:])
            break
    
    # Extract avalanche effect if present
    avalanche_effect = None
    for part in parts:
        if part.startswith("aw"):
            avalanche_effect = float(part[2:])
            break
    
    pbox = None
    for part in parts:
        if part.startswith("p"):
            pbox = part[1:]
            break
    
    return {
        "rounds": rounds,
        "nl": nl,
        "sbox_name": NONLINEARITY_MAP[nl],
        "hidden_size": hidden_size,
        "avalanche_effect": avalanche_effect,
        "pbox": pbox,
        "is_random": is_random,
        "plaintext_type": "random" if is_random else "text"
    }


def decrypt_csv(filepath: str, key: bytes, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Decrypt predictions in a CSV file and compute comprehensive metrics.
    
    Args:
        filepath: Path to input CSV file
        key: Encryption key
        verbose: Print progress messages
        
    Returns:
        Tuple of (enriched_dataframe, metrics_dict)
        
    The enriched dataframe includes:
        - Original columns
        - decrypted_predicted: Decrypted predicted plaintext
        - decrypted_actual: Decrypted actual plaintext
        - cosine_similarity: Cosine similarity between prediction and actual
        - per_sample_bit_accuracy: Bit-level accuracy for this sample
        
    The metrics dict includes:
        - All parameters from parse_filename
        - avg_cos: Average cosine similarity
        - std_cos: Standard deviation of cosine similarity
        - bit_acc: Overall bit-level accuracy from file header
        - avg_bit_acc: Average of per-sample bit accuracies
        - word_acc: Word-level accuracy (exact match percentage)
    """
    filename = os.path.basename(filepath)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
    
    # Parse file parameters
    params = parse_filename(filename)
    curr_sbox = SBOXES[params["sbox_name"]]
    
    if verbose:
        print(f"Configuration:")
        print(f"  Rounds: {params['rounds']}")
        print(f"  Nonlinearity: {params['nl']} ({params['sbox_name']})")
        print(f"  Hidden Size: {params['hidden_size']}")
        print(f"  Plaintext Type: {params['plaintext_type']}")
    
    # Initialize cipher
    
    # Read bit-level accuracy from row 2
    with open(filepath, "r") as f:
        lines = f.readlines()
        bit_acc = None
        if len(lines) > 1 and "Bit-Level Accuracy" in lines[1]:
            bit_acc = float(lines[1].strip().split(":")[1].replace("%", ""))
    
    # Read main data (header on line 3)
    if "aw" in filename:
        df = pd.read_csv(filepath, header=7)
    else:
        df = pd.read_csv(filepath, header=2)

    print("Reading file:", filepath)
    print("\n\n\n\n")
    
    # Lists to store computed values
    predicted_msgs = []
    actual_msgs = []
    cosines = []
    bit_accuracies = []
    
    # Process each row
    for idx, row in df.iterrows():
        # need to determine the P-box
        if params["pbox"] == "Default":
            pbox = pbox_default
            pbox_inv = pbox_default_inv
        elif params["pbox"] == "Trivial":
            pbox = pbox_trivial
            pbox_inv = pbox_trivial_inv
        elif params["pbox"] == "Weak":
            pbox = pbox_weak
            pbox_inv = pbox_weak_inv
        cipher = Present(key, rounds=params["rounds"], sbox=curr_sbox, pbox=pbox, pbox_inv=pbox_inv)

        # Parse prediction probabilities
        pred_probs = np.array(ast.literal_eval(row["prediction"]), dtype=float)
        actual_bits = np.array(list(map(int, row["actual_bits"])), dtype=float)
        
        # Ensure equal length
        n = min(len(pred_probs), len(actual_bits))
        v_pred, v_actual = pred_probs[:n], actual_bits[:n]
        
        # Calculate cosine similarity
        cos = np.dot(v_pred, v_actual) / (np.linalg.norm(v_pred) * np.linalg.norm(v_actual))
        cosines.append(float(cos))
        
        # Calculate per-sample bit accuracy
        pred_bits_binary = (v_pred >= 0.5).astype(int)
        per_sample_acc = (pred_bits_binary == v_actual).mean() * 100
        bit_accuracies.append(per_sample_acc)
        
        # Convert probabilities to bits (threshold at 0.5)
        pred_bits = ''.join('1' if p >= 0.5 else '0' for p in v_pred)
        pred_bytes = bits_to_bytes(pred_bits)
        actual_bytes = bits_to_bytes(row["actual_bits"])
        
        # Decrypt both predicted and actual
        decrypted_pred = cipher.decrypt(pred_bytes).decode(errors="ignore")
        decrypted_actual = cipher.decrypt(actual_bytes).decode(errors="ignore")
        
        # Clean: keep only English letters
        if "random" not in filepath:
            decrypted_pred = re.sub(r'[^A-Za-z]', '?', decrypted_pred)
            decrypted_actual = re.sub(r'[^A-Za-z]', '?', decrypted_actual)

            # print(f"Decrypted prediction: {decrypted_pred}")
            # print(f"Decrypted actual: {decrypted_actual}")

        predicted_msgs.append(decrypted_pred)
        actual_msgs.append(decrypted_actual)
    
    # Add new columns to dataframe
    df["decrypted_predicted"] = predicted_msgs
    df["decrypted_actual"] = actual_msgs
    df["cosine_similarity"] = cosines
    df["per_sample_bit_accuracy"] = bit_accuracies
    
    # Calculate aggregate metrics
    avg_cos = np.mean(cosines)
    std_cos = np.std(cosines)
    avg_bit_acc = np.mean(bit_accuracies)
    
    # Calculate word-level accuracy (exact matches)
    word_matches = sum(1 for p, a in zip(predicted_msgs, actual_msgs) if p == a)
    word_acc = (word_matches / len(predicted_msgs)) * 100 if predicted_msgs else 0
    
    if verbose:
        print(f"\nMetrics:")
        print(f"  Average Cosine Similarity: {avg_cos:.4f} ± {std_cos:.4f}")
        print(f"  Average Bit Accuracy: {avg_bit_acc:.2f}%")
        print(f"  Word-Level Accuracy: {word_acc:.2f}% ({word_matches}/{len(predicted_msgs)})")
        
        # Show sample discrepancies for low rounds/nonlinearity
        if params["nl"] < 2 and params["rounds"] < 5:
            discrepancies = [(p, a) for p, a in zip(predicted_msgs, actual_msgs) if p != a]
            if discrepancies:
                print(f"\n  Sample discrepancies ({len(discrepancies)} total):")
                for p, a in discrepancies[:3]:  # Show first 3
                    print(f"    Predicted: '{p}' | Actual: '{a}'")
    
    # Compile metrics dictionary
    metrics = {
        **params,
        "avg_cos": avg_cos,
        "std_cos": std_cos,
        "bit_acc": bit_acc if bit_acc is not None else avg_bit_acc,
        "avg_bit_acc": avg_bit_acc,
        "word_acc": word_acc,
        "total_samples": len(df)
    }
    
    return df, metrics


def decrypt_all_csvs(results_dir: str, key: bytes, clean: bool = True) -> Tuple[List[Dict], pd.DataFrame]:
    """Decrypt all CSV files in results directory.
    
    Args:
        results_dir: Directory containing CSV files
        key: Encryption key
        clean: Remove existing decrypted files before processing
        
    Returns:
        Tuple of (metrics_list, summary_dataframe)
        - metrics_list: List of metric dictionaries for each file
        - summary_dataframe: DataFrame with all metrics for easy analysis
    """
    print("\n" + "="*60)
    print("DECRYPTION AND ANALYSIS")
    print("="*60)
    
    # Clean existing decrypted files
    if clean:
        print("\nCleaning existing decrypted files...")
        removed_count = 0
        for filename in os.listdir(results_dir):
            if "decrypted" in filename:
                file_path = os.path.join(results_dir, filename)
                os.remove(file_path)
                removed_count += 1
        print(f"Removed {removed_count} existing decrypted file(s)")
    
    # Process all CSV files
    all_metrics = []
    
    csv_files = [f for f in os.listdir(results_dir) 
                 if f.endswith(".csv") and "decrypted" not in f]
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process")
    
    for i, filename in enumerate(csv_files, 1):
        if "decryption_summary" in filename:
            continue
        else:
            print(f"\n[{i}/{len(csv_files)}]", end=" ")
            
            filepath = os.path.join(results_dir, filename)
            
            # Decrypt and analyze
            df, metrics = decrypt_csv(filepath, key, verbose=True)
            
            # Save decrypted version
            out_path = os.path.join(results_dir, f"decrypted_{filename}")
            df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
            
            all_metrics.append(metrics)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(all_metrics)
    
    # Save summary
    summary_path = os.path.join(results_dir, "decryption_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(all_metrics)}")
    print(f"Summary saved to: {summary_path}")
    print("\nSummary statistics:")
    print(summary_df[["rounds", "nl", "plaintext_type", "bit_acc", "word_acc"]].describe())
    
    return all_metrics, summary_df


def main():
    """Main execution function."""
    # Initialize key
    random.seed(RANDOM_SEED)
    k = f"{random.getrandbits(80):020x}"
    print("="*60)
    print("CIPHER DECRYPTION TOOL")
    print("="*60)
    print(f"Using key: {k}")
    print(f"Results directory: {RESULTS_DIR}")
    
    key = bytes.fromhex(k)
    
    # Decrypt all CSVs
    metrics_list, summary_df = decrypt_all_csvs(RESULTS_DIR, key, clean=CLEAN_EXISTING)
    
    print("\n" + "="*60)
    print("DECRYPTION COMPLETE")
    print("="*60)
    print(f"\nDecrypted files are prefixed with 'decrypted_'")
    print(f"Summary metrics available in 'decryption_summary.csv'")
    print(f"\nUse the visualization script to generate plots from these results.")


if __name__ == "__main__":
    main()