import torch
from torch import nn
import pandas as pd
import csv
import json
import os
import glob

rounds = [1,2,3,4,5,6,7,8,9,10,16,32]
hidden_sizes = [128,256,512,1024]

prefixes = ["", "random_"]  # run both normal and random datasets

# Create results directory
os.makedirs("./new_results", exist_ok=True)

for hidden_size in hidden_sizes:
    for curr_round in rounds:
        for prefix in prefixes:
            # Find all CSV files matching the pattern for this round and prefix
            pattern = f"./new_data/{prefix}{curr_round}_rounds_nl*_aw*_bits.csv"
            matching_files = sorted(glob.glob(pattern))
            
            if not matching_files:
                print(f"⚠️  No files found for pattern: {pattern}")
                continue
            
            round_scores = []
            config_info = []
            
            for data_path in matching_files:
                # Extract nl and aw from filename
                # Format: {prefix}{rounds}_rounds_nl{nl}_aw{aw}_bits.csv
                filename = os.path.basename(data_path)
                parts = filename.replace(prefix, "").replace("_bits.csv", "").split("_")
                
                nl_str = [p for p in parts if p.startswith("nl")][0]
                aw_str = [p for p in parts if p.startswith("aw")][0]
                
                curr_nl = int(nl_str[2:])
                curr_aw = float(aw_str[2:])
                
                print(f"\n{'='*80}")
                print(f"Loading dataset: {data_path}")
                print(f"Rounds: {curr_round}, NL: {curr_nl}, AW: {curr_aw}, Prefix: {prefix or 'normal'}")
                print(f"{'='*80}")
                
                # === Load dataset ===
                df = pd.read_csv(data_path)

                X = torch.tensor([[int(b) for b in bits] for bits in df["plaintext_bits"]], dtype=torch.float32)
                Y = torch.tensor([[int(b) for b in bits] for bits in df["ciphertext_bits"]], dtype=torch.float32)

                print("Full dataset shape:", X.shape, Y.shape)

                # === Split into train (900) and test (100) ===
                num_samples = X.size(0)
                train_size = 900
                X_train, Y_train = X[:train_size], Y[:train_size]
                X_test, Y_test = X[train_size:], Y[train_size:]

                # === Define neural network ===
                class NeuralNetwork(nn.Module):
                    def __init__(self, input_size=64, hidden_size=128, output_size=64, num_rounds=1):
                        super().__init__()
                        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
                        for _ in range(num_rounds):
                            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
                        layers += [nn.Linear(hidden_size, output_size), nn.Sigmoid()]
                        self.layers = nn.Sequential(*layers)

                    def forward(self, x):
                        return self.layers(x)

                print(f"Building model with {curr_round} layers, hidden={hidden_size}")
                model = NeuralNetwork(hidden_size=hidden_size, num_rounds=curr_round)
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                # === Training loop ===
                num_epochs = 20
                batch_size = 20
                for epoch in range(num_epochs):
                    perm = torch.randperm(train_size)
                    epoch_loss = 0.0
                    for i in range(0, train_size, batch_size):
                        idx = perm[i:i+batch_size]
                        bx, by = X_train[idx], Y_train[idx]
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    if (epoch+1) % 5 == 0:
                        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/train_size:.6f}")

                # === Evaluation ===
                with torch.no_grad():
                    preds = model(X_test)
                    preds_rounded = (preds > 0.5).float()
                    test_loss = criterion(preds, Y_test).item()
                    bit_accuracy = (preds_rounded == Y_test).float().mean().item()

                print(f"✓ Test Loss: {test_loss:.6f}, Bit Acc: {bit_accuracy*100:.2f}%")

                # === Save results ===
                filename = f"./new_results/{prefix}r{curr_round}_nl{curr_nl}_aw{curr_aw}_hs{hidden_size}.csv"
                with open(filename, "w", newline="") as f:
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
                    for i in range(len(X_test)):
                        predictions = json.dumps(preds[i].tolist())
                        pred_bits = "".join(str(int(b)) for b in preds_rounded[i].tolist())
                        actual_bits = "".join(str(int(b)) for b in Y_test[i].tolist())
                        writer.writerow([predictions, pred_bits, actual_bits])
                print(f"✓ Saved {len(X_test)} predictions to {filename}")

                round_scores.append((curr_nl, curr_aw, bit_accuracy))
                config_info.append(f"nl={curr_nl}, aw={curr_aw}")

            # === Compare configurations ===
            if len(round_scores) > 1:
                print(f"\n{'='*80}")
                print(f"Security Analysis for Round {curr_round} ({prefix or 'normal'} dataset, hs={hidden_size})")
                print(f"{'='*80}")
                
                # Sort by accuracy (lower accuracy = more secure)
                sorted_configs = sorted(round_scores, key=lambda x: x[2])
                
                print("\nRanking (most secure to least secure):")
                for i, (nl, aw, acc) in enumerate(sorted_configs, 1):
                    print(f"  {i}. NL={nl}, AW={aw:.2f} → Accuracy: {acc*100:.2f}%")
                
                # Pairwise comparisons
                print("\nPairwise Security Comparisons:")
                for i in range(len(round_scores)):
                    for j in range(i+1, len(round_scores)):
                        nl1, aw1, acc1 = round_scores[i]
                        nl2, aw2, acc2 = round_scores[j]
                        
                        if acc1 < acc2:
                            margin = (acc2 - acc1) * 100
                            print(f"  • (NL={nl1}, AW={aw1:.2f}) is MORE secure than (NL={nl2}, AW={aw2:.2f}) by {margin:.2f}% accuracy")
                        elif acc1 > acc2:
                            margin = (acc1 - acc2) * 100
                            print(f"  • (NL={nl2}, AW={aw2:.2f}) is MORE secure than (NL={nl1}, AW={aw1:.2f}) by {margin:.2f}% accuracy")
                
                print(f"{'='*80}\n")

print("\n" + "="*80)
print("All training complete!")
print("="*80)