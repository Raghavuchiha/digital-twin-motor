import os
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Get CSV file path from the user
csv_file = "D:\\sem-7\\mini-project\\clean_motor.csv"  # Replace with your CSV file path

data = pd.read_csv(csv_file)

print("CSV columns:", data.columns.tolist())
print(data.head())

output_folder = "outputs_stage2"
os.makedirs(output_folder, exist_ok=True)
os.makedirs("models", exist_ok=True)


# Features used for fault detection
features = [
    "Current",
    "Voltage",
    "temp_mot",
    "1x_Vibration",
    "5x_Vibration",
    "temp_mot_d5",
    "vib1x_d5"
]


# Train the model using normal motor data
def train_model(data):
    normal_data = data[data["label"] == "normal"]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(normal_data[features])

    model = IsolationForest(
        n_estimators=200,
        random_state=42
    )
    model.fit(scaled_data)

    # Calculate the anomaly threshold
    scores = -model.score_samples(scaled_data)
    threshold = np.percentile(scores, 99)

    return scaler, model, threshold


# Calculate anomaly scores
def get_scores(scaler, model, data):
    scaled_data = scaler.transform(data[features])
    return -model.score_samples(scaled_data)


# Evaluate the model
def evaluate_model(data, scores, threshold, title):
    flagged = scores > threshold

    print(f"\n=== {title} ===")
    print(f"Threshold: {threshold:.3f}")

    for label in ["normal", "stress", "failed"]:
        rows = data["label"].values == label

        if rows.sum() > 0:
            total = rows.sum()
            abnormal = flagged[rows].sum()
            percentage = abnormal / total * 100

            print(
                f"{label}: {abnormal}/{total} samples flagged "
                f"({percentage:.1f}%)"
            )


# Create and save the anomaly plot
def plot_results(data, scores, threshold, title, filename):
    colors = {
        "normal": "#c8e6c9",
        "stress": "#ffe0b2",
        "failed": "#ffcdd2"
    }

    fig, ax = plt.subplots(figsize=(11, 4))

    labels = data["label"].values
    x = np.arange(len(data))

    # Add background colours for each motor condition
    start = 0

    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            label = labels[start]

            ax.axvspan(
                start,
                i - 1,
                color=colors[label],
                alpha=0.8,
                label=label
            )

            start = i

    ax.plot(x, scores, color="black", label="Anomaly score")
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Threshold"
    )

    ax.set_title(title)
    ax.set_xlabel("Recording number")
    ax.set_ylabel("Anomaly score")

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, filename), dpi=130)
    plt.close(fig)


def main():
    # Load the dataset
    data = pd.read_csv(csv_file)

    run1 = data[data["run"] == 1].reset_index(drop=True)
    run2 = data[data["run"] == 2].reset_index(drop=True)

    # Test A: Train on run 1, test on run 2
    scaler, model, threshold = train_model(run1)
    scores = get_scores(scaler, model, run2)

    evaluate_model(
        run2, scores, threshold,
        "Test A: Train run 1, test run 2"
    )

    plot_results(
        run2, scores, threshold,
        "Test A: Train run 1, test run 2",
        "test_A_run2.png"
    )

    # Test B: Train on run 2, test on run 1
    scaler, model, threshold = train_model(run2)
    scores = get_scores(scaler, model, run1)

    evaluate_model(
        run1, scores, threshold,
        "Test B: Train run 2, test run 1"
    )

    plot_results(
        run1, scores, threshold,
        "Test B: Train run 2, test run 1",
        "test_B_run1.png"
    )

    # Train the final model using normal data from both runs
    scaler, model, threshold = train_model(data)

    # Save the model
    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "threshold": threshold,
            "features": features
        },
        "models/fault_model.joblib"
    )

    print("\nModel saved: models/fault_model.joblib")
    print(f"Threshold: {threshold:.3f}")
    print(f"Plots saved in: {output_folder}/")


if __name__ == "__main__":
    main()