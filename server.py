# server.py
from pathlib import Path
import flwr as fl
import torch  # needed to instantiate the model and read state_dict

# Build the same model as clients to create initial parameters
from model_utils import ParkinsonsNetSingle as ParkinsonsNet  # single-target net

FEATURES_FILE = Path("clients/_features.txt")


# ------------------------------
# Metrics aggregators (Flower 1.22 -> results are two-tuples)
# ------------------------------
def aggregate_eval_metrics(results):
    """
    results: List[Tuple[num_examples:int, metrics:dict]] from evaluate()
    Returns weighted average per metric key.
    """
    if not results:
        return {}
    total = sum(num for num, _ in results)
    agg = {}
    for num, metrics in results:
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + num * float(v)
    for k in agg:
        agg[k] /= total
    print(f"[AGG EVAL] {agg}")
    return agg


def aggregate_fit_metrics(results):
    """
    results: List[Tuple[num_examples:int, metrics:dict]] from fit()
    Returns weighted average per metric key.
    """
    if not results:
        return {}
    total = sum(num for num, _ in results)
    agg = {}
    for num, metrics in results:
        if not metrics:
            continue
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + num * float(v)
    if total > 0:
        for k in agg:
            agg[k] /= total
    print(f"[AGG FIT] {agg}")
    return agg


# ------------------------------
# Create initial global parameters for FedAdam (required in flwr 1.22.0)
# ------------------------------
def make_initial_parameters():
    # read feature list to know input size
    feats = [ln.strip() for ln in FEATURES_FILE.read_text().splitlines() if ln.strip()]
    in_features = len(feats)

    # build same model as clients and take its initial weights
    model = ParkinsonsNet(in_features=in_features)
    ndarrays = [p.detach().cpu().numpy() for _, p in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(ndarrays)


# ------------------------------
# Strategy: FedAdam (server-side adaptive optimizer)
# ------------------------------
from flwr.server.strategy import FedAdam

strategy = FedAdam(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    # gentler server updates
    eta=0.003,      # was 0.01
    eta_l=0.0005,   # was 0.001
    beta_1=0.9,
    beta_2=0.99,
    tau=0.0,
    evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    initial_parameters=make_initial_parameters(),
)


def main():
    print("🚀 Starting Flower server on 127.0.0.1:8080 ...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=25),
    )


if __name__ == "__main__":
    main()
