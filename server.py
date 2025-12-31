import matplotlib.pyplot as plt
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

# ---------------------------
# Globals to store history
# ---------------------------
history_loss = []
history_acc = []

# ---------------------------
# Metric aggregation
# ---------------------------
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# ---------------------------
# Custom Strategy
# ---------------------------
class PlottingStrategy(FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if loss is not None:
            history_loss.append(loss)

        if metrics is not None and "accuracy" in metrics:
            history_acc.append(metrics["accuracy"])

        print(f"Round {server_round} | Loss: {loss}, Acc: {metrics}")

        return loss, metrics

# ---------------------------
# Server function
# ---------------------------
def server_fn(context):
    num_rounds = context.run_config.get("num-rounds", 5)

    strategy = PlottingStrategy(
        evaluate_metrics_aggregation_fn=weighted_average,
        fraction_evaluate=1.0,
        min_evaluate_clients=2,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# ---------------------------
# Run server
# ---------------------------
app = ServerApp(server_fn=server_fn)

# ---------------------------
# Plot AFTER training ends
# ---------------------------
def plot_results():
    rounds = range(1, len(history_loss) + 1)

    plt.figure(figsize=(8, 5))

    # Loss (left axis)
    plt.plot(
        rounds,
        history_loss,
        label="Loss",
        color="#d62728",      # deep red
        linewidth=2.5,
        marker="o",
        markersize=6
    )

    # Accuracy (right axis)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(
        rounds,
        history_acc,
        label="Accuracy",
        color="#1f77b4",      # clean blue
        linewidth=2.5,
        marker="s",
        markersize=6
    )

    # Axis labels
    ax1.set_xlabel("Federated Rounds", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11, color="#d62728")
    ax2.set_ylabel("Accuracy", fontsize=11, color="#1f77b4")

    # Grid (only on loss axis for clarity)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Title
    plt.title("Federated Training Performance", fontsize=13, fontweight="bold")

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="center right"
    )

    plt.tight_layout()
    plt.savefig("fl_results.png", dpi=300)
    print("Saved plot to fl_results.png")


# This runs ONLY when the process ends
import atexit
atexit.register(plot_results)

