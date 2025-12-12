
import torch
from torch import nn
import pandas as pd
import csv
import json
import os
import glob
import torch.multiprocessing as mp
import math

# Configuration
rounds = [1,2,3,4,5,6,7,8,9,10,16,32]
hidden_sizes = [128,256,512,1024]
prefixes = ["", "random_"]
DATA_DIR = "./data" # Updated from new_data
RESULTS_DIR = "./new_results"

def train_model(gpu_id, tasks, queue):
    try:
        # Assign device for this process
        if torch.cuda.is_available():
            dev_idx = gpu_id % torch.cuda.device_count()
            device = torch.device(f"cuda:{dev_idx}")
        else:
            device = torch.device("cpu")
            
        print(f"Process {gpu_id} using device: {device}")

        for task in tasks:
            hidden_size, curr_round, prefix = task
            
            # Find all CSV files matching the pattern
            pattern = f"{DATA_DIR}/{prefix}{curr_round}_rounds_nl*_aw*_bits.csv"
            matching_files = sorted(glob.glob(pattern))
            
            if not matching_files:
                # print(f"⚠️  No files found for pattern: {pattern}")
                continue
                
            round_scores = []
            
            for data_path in matching_files:
                try:
                    # Extract nl and aw from filename
                    filename = os.path.basename(data_path)
                    parts = filename.replace(prefix, "").replace("_bits.csv", "").split("_")
                    
                    nl_str = [p for p in parts if p.startswith("nl")][0]
                    aw_str = [p for p in parts if p.startswith("aw")][0]
                    
                    curr_nl = int(nl_str[2:])
                    # Handle aw which might have decimal points or not, but filename format is usually consistent
                    # If aw is like aw0.5, it formats as aw0.5. If aw1, aw1.
                    # The original code did: float(aw_str[2:])
                    curr_aw = float(aw_str[2:])
                    
                    print(f"[GPU {gpu_id}] Processing: R{curr_round}, HS{hidden_size}, {prefix or 'normal'}, NL{curr_nl}, AW{curr_aw}")
                    
                    # === Load dataset ===
                    df = pd.read_csv(data_path)
                    X = torch.tensor([[int(b) for b in bits] for bits in df["plaintext_bits"]], dtype=torch.float32).to(device)
                    Y = torch.tensor([[int(b) for b in bits] for bits in df["ciphertext_bits"]], dtype=torch.float32).to(device)
                    
                    # Split
                    num_samples = X.size(0)
                    train_size = 900
                    if num_samples < train_size: 
                        train_size = int(num_samples * 0.9)
                        
                    X_train, Y_train = X[:train_size], Y[:train_size]
                    X_test, Y_test = X[train_size:], Y[train_size:]
                    
                    # === Define neural network ===
                    layers = [nn.Linear(64, hidden_size), nn.ReLU()]
                    for _ in range(curr_round):
                        layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
                    layers += [nn.Linear(hidden_size, 64), nn.Sigmoid()]
                    model = nn.Sequential(*layers).to(device)
                    
                    criterion = nn.BCELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    
                    # === Training loop ===
                    model.train()
                    num_epochs = 20
                    batch_size = 20
                    for epoch in range(num_epochs):
                        perm = torch.randperm(train_size, device=device)
                        for i in range(0, train_size, batch_size):
                            idx = perm[i:i+batch_size]
                            bx, by = X_train[idx], Y_train[idx]
                            optimizer.zero_grad()
                            out = model(bx)
                            loss = criterion(out, by)
                            loss.backward()
                            optimizer.step()
                            
                    # === Evaluation ===
                    model.eval()
                    with torch.no_grad():
                        preds = model(X_test)
                        preds_rounded = (preds > 0.5).float()
                        test_loss = criterion(preds, Y_test).item()
                        bit_accuracy = (preds_rounded == Y_test).float().mean().item()
                    
                    # === Save results ===
                    out_filename = f"{RESULTS_DIR}/{prefix}r{curr_round}_nl{curr_nl}_aw{curr_aw}_hs{hidden_size}.csv"
                    with open(out_filename, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"Rounds: {curr_round}"])
                        writer.writerow([f"Nonlinearity: {curr_nl}"])
                        writer.writerow([f"Avalanche Weight: {curr_aw}"])
                        writer.writerow([f"Hidden Size: {hidden_size}"])
                        writer.writerow([f"Prefix: {prefix or 'normal'}"])
                        writer.writerow([f"Test Loss: {test_loss}"])
                        writer.writerow([f"Bit-Level Accuracy: {bit_accuracy*100:.2f}%"])
                        writer.writerow([])
                        writer.writerow(["prediction", "predicted_bits", "actual_bits"])
                        
                        preds_cpu = preds.cpu()
                        preds_rounded_cpu = preds_rounded.cpu()
                        Y_test_cpu = Y_test.cpu()
                        
                        for i in range(len(X_test)):
                            predictions = json.dumps(preds_cpu[i].tolist())
                            pred_bits = "".join(str(int(b)) for b in preds_rounded_cpu[i].tolist())
                            actual_bits = "".join(str(int(b)) for b in Y_test_cpu[i].tolist())
                            writer.writerow([predictions, pred_bits, actual_bits])
                            
                    round_scores.append((curr_nl, curr_aw, bit_accuracy))
                    
                except Exception as e:
                    print(f"Error processing {data_path} on GPU {gpu_id}: {e}")
                    
            # === Security Analysis ===
            if len(round_scores) > 1:
                # Capture the analysis output to print it safely
                output = []
                output.append(f"\n{'='*80}")
                output.append(f"Security Analysis for Round {curr_round} ({prefix or 'normal'} dataset, hs={hidden_size})")
                output.append(f"{'='*80}")
                
                sorted_configs = sorted(round_scores, key=lambda x: x[2])
                
                output.append("\nRanking (most secure to least secure):")
                for i, (nl, aw, acc) in enumerate(sorted_configs, 1):
                    output.append(f"  {i}. NL={nl}, AW={aw:.2f} → Accuracy: {acc*100:.2f}%")
                
                output.append("\nPairwise Security Comparisons:")
                for i in range(len(round_scores)):
                    for j in range(i+1, len(round_scores)):
                        nl1, aw1, acc1 = round_scores[i]
                        nl2, aw2, acc2 = round_scores[j]
                        
                        if acc1 < acc2:
                            margin = (acc2 - acc1) * 100
                            output.append(f"  • (NL={nl1}, AW={aw1:.2f}) is MORE secure than (NL={nl2}, AW={aw2:.2f}) by {margin:.2f}% accuracy")
                        elif acc1 > acc2:
                            margin = (acc1 - acc2) * 100
                            output.append(f"  • (NL={nl2}, AW={aw2:.2f}) is MORE secure than (NL={nl1}, AW={aw1:.2f}) by {margin:.2f}% accuracy")
                
                output.append(f"{'='*80}\n")
                print("\n".join(output))

    except Exception as e:
        print(f"Worker {gpu_id} crashed: {e}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate all tasks
    all_tasks = []
    for hidden_size in hidden_sizes:
        for curr_round in rounds:
            for prefix in prefixes:
                all_tasks.append((hidden_size, curr_round, prefix))
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. Using CPU.")
        num_gpus = 1
    else:
        print(f"Found {num_gpus} GPUs. Distributing {len(all_tasks)} tasks.")

    # Distribute tasks
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        tasks_per_gpu[i % num_gpus].append(task)
        
    processes = []
    queue = mp.Queue()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=train_model, args=(gpu_id, tasks_per_gpu[gpu_id], queue))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("\n" + "="*80)
    print("All distributed training complete!")
    print("="*80)