import torch
from torch import nn
import pandas as pd
import csv
import json

rounds = [1,2,3,4,5,6,7,8,9,10,16,32]
# rounds = [8]
nl = [0,2,4]
hidden_sizes = [128,256,512,1024]

prefixes = [""]  # run both normal and random datasets

for hidden_size in hidden_sizes:
    for curr_round in rounds:
        for prefix in prefixes:
            round_scores = []
            for curr_nl in nl:
                # === Load dataset ===
                data_path = f"./data/{prefix}{curr_round}_rounds_nl{curr_nl}_bits.csv"
                print(f"\nLoading dataset: {data_path}")
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

                print(f"Building model with {curr_round} layers, hidden={hidden_size}, prefix={prefix or 'normal'}, nl={curr_nl}")
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

                print(f"Test Loss: {test_loss:.6f}, Bit Acc: {bit_accuracy*100:.2f}%")

                # === Save results ===
                filename = f"./results/{prefix}r{curr_round}_nl{curr_nl}_hs{hidden_size}.csv"
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"Test Loss: {test_loss}"])
                    writer.writerow([f"Bit-Level Accuracy: {bit_accuracy*100:.2f}%"])
                    writer.writerow(["prediction", "predicted_bits", "actual_bits"])
                    for i in range(len(X_test)):
                        predictions = json.dumps(preds[i].tolist())
                        pred_bits = "".join(str(int(b)) for b in preds_rounded[i].tolist())
                        actual_bits = "".join(str(int(b)) for b in Y_test[i].tolist())
                        writer.writerow([predictions, pred_bits, actual_bits])
                print(f"Saved {len(X_test)} predictions to {filename}")

                round_scores.append(bit_accuracy)

            for i in range(1, 3):
                if round_scores[i-1] < round_scores[i]:
                    print(f"Round {curr_round}: nl {2*(i-1)} more secure than nl {2*i} ({prefix or 'normal'})")
