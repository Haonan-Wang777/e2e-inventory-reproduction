"""Runnable E2E mechanism demonstration; this does not retrain the paper's model.

Only the five released sales histories are observed data. Order timing, lead time,
and starting stock are explicitly constructed scenarios. See config.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def optimum_order(demand, inventory_at_arrival, b=9.0, h=1.0):
    """Theorem 1, inclusive s*: select upper minimizer when b*n/(b+h) is integer."""
    d = np.asarray(demand, dtype=float)
    if d.ndim != 1 or len(d) == 0 or (d < 0).any() or b <= 0 or h <= 0:
        raise ValueError("Demand must be a nonempty nonnegative vector; b,h > 0")
    index = min(int(math.floor(b * len(d) / (b + h))), len(d) - 1)
    return max(float(np.cumsum(d)[index] - inventory_at_arrival), 0.0)


def interval_cost(order, demand, inventory_at_arrival, b=9.0, h=1.0):
    inventory = inventory_at_arrival + order - np.cumsum(demand)
    holding = h * np.maximum(inventory, 0).sum()
    shortage = b * np.maximum(-inventory, 0).sum()
    return float(holding), float(shortage)


def verify_label_formula(seed=4564, cases=250):
    """Independent candidate enumeration, including zero-order and tie cases."""
    rng = np.random.default_rng(seed)
    worst_gap = 0.0
    for _ in range(cases):
        d = rng.integers(0, 12, size=int(rng.integers(1, 18)))
        stock = int(rng.integers(-10, 45))
        b, h = int(rng.integers(1, 12)), int(rng.integers(1, 6))
        q = optimum_order(d, stock, b, h)
        # A convex piecewise-linear function reaches a minimum at a kink or q=0.
        candidates = np.unique(np.r_[0.0, np.maximum(np.cumsum(d) - stock, 0.0)])
        brute = min(sum(interval_cost(c, d, stock, b, h)) for c in candidates)
        gap = abs(sum(interval_cost(q, d, stock, b, h)) - brute)
        worst_gap = max(worst_gap, gap)
        if gap > 1e-8:
            raise AssertionError((d, stock, b, h, q, gap))
    return {"cases": cases, "maximum_objective_gap": worst_gap, "passed": True}


class MQRNN(nn.Module):
    """Shared feature encoder, LSTM, global/local decoders and a separate bridge."""
    def __init__(self, config):
        super().__init__()
        self.horizon = config["prediction_days"]
        self.context = config["context_size"]
        self.quantile_count = len(config["quantiles"])
        self.encoder = nn.Linear(2, self.context)
        self.lstm = nn.LSTM(self.context + 1, config["hidden_size"], batch_first=True)
        self.global_mlp = nn.Linear(config["hidden_size"] + self.context * self.horizon,
                                    self.context * (self.horizon + 1))
        self.local_mlp = nn.Linear(self.context * 3, self.quantile_count)
        self.bridge_mlp = nn.Linear(self.context * 3, config["bridge_size"])

    def forward(self, hist_x, hist_y, future_x):
        history = torch.tanh(self.encoder(hist_x))
        future = torch.tanh(self.encoder(future_x))
        _, (h, _) = self.lstm(torch.cat([history, hist_y.unsqueeze(-1)], dim=-1))
        context = torch.sigmoid(self.global_mlp(torch.cat([h[-1], future.flatten(1)], dim=1)))
        local_context = context[:, : self.horizon * self.context].reshape(-1, self.horizon, self.context)
        shared_context = context[:, self.horizon * self.context :].unsqueeze(1).expand(-1, self.horizon, -1)
        local_input = torch.cat([local_context, shared_context, future], dim=-1)
        return self.local_mlp(local_input), self.bridge_mlp(local_input).flatten(1)


class E2EDemo(nn.Module):
    def __init__(self, config, sku_count):
        super().__init__()
        dropout = config["dropout"]
        self.demand = MQRNN(config)
        self.lead_hidden = nn.Sequential(nn.Linear(2 + sku_count, 50), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(50, 20), nn.ReLU(), nn.Dropout(dropout))
        self.lead_output = nn.Linear(20, 1)
        integration_size = 20 + config["prediction_days"] * config["bridge_size"] + 2
        self.integration = nn.Sequential(nn.Linear(integration_size, 100), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(100, 100), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(100, 1), nn.Softplus())
        for name, parameter in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(parameter, mean=0.0, std=0.01)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, hist_x, hist_y, future_x, lead_x, cat_x, state_x):
        forecast, bridge = self.demand(hist_x, hist_y, future_x)
        lead_hidden = self.lead_hidden(torch.cat([lead_x, cat_x], dim=1))
        decision = self.integration(torch.cat([lead_hidden, bridge, state_x], dim=1)).squeeze(-1)
        return decision, self.lead_output(lead_hidden).squeeze(-1), forecast


def calendar_features(days):
    return np.stack([np.sin(2 * np.pi * days / 7), np.cos(2 * np.pi * days / 7)], axis=-1)


def build_data(config):
    df = pd.read_csv(ROOT / "author_package" / "E2E-model-code" / "Data" / "rdc_sales_1320.csv", index_col=0)
    sales = df.to_numpy(dtype=np.float64)
    if sales.shape != (5, 457) or not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Unexpected released sales data; inspect provenance before proceeding")
    n_sku, n_days = sales.shape
    train_end, val_end = int(n_days * .6), int(n_days * .8)
    log_scale = max(float(np.log1p(sales[:, :train_end]).max()), 1.0)
    train_mean_scale = max(float(sales[:, :train_end].mean()), 1.0)
    calendar = calendar_features(np.arange(n_days)).astype(np.float32)
    datasets, metadata = {}, {}
    history, prediction = config["history_days"], config["prediction_days"]
    lead, review = config["assumed_lead_days"], config["assumed_review_days"]
    boundaries = {"train": (history, train_end), "validation": (train_end, val_end), "test": (val_end, n_days)}
    for name, (start, end) in boundaries.items():
        rows, meta = [], []
        for sku in range(n_sku):
            for t in range(start, end - prediction + 1):
                # Hypothetical daily order scenario; no released stock or order ID is joined.
                past_mean = float(sales[sku, t-28:t].mean())
                factor = .3 + .9 * ((t + sku * 7) % 29) / 28
                stock_at_order = past_mean * (lead + review) * factor
                prearrival_demand = float(sales[sku, t:t+lead].sum())
                stock_at_arrival = stock_at_order - prearrival_demand
                arrival_demand = sales[sku, t+lead:t+lead+review]
                q = optimum_order(arrival_demand, stock_at_arrival, config["b"], config["h"])
                rows.append((calendar[t-history:t], np.log1p(sales[sku,t-history:t])/log_scale,
                             calendar[t:t+prediction], np.array([lead/10, 0.0]), np.eye(n_sku)[sku],
                             np.array([stock_at_order, review/30]), q, lead/10,
                             np.log1p(sales[sku,t:t+prediction])/log_scale))
                meta.append({"sku": str(df.index[sku]), "origin_day": t, "last_future_day": t+prediction-1,
                             "stock_at_order": stock_at_order, "stock_at_arrival": stock_at_arrival,
                             "oracle_order": q, "arrival_demand": arrival_demand.tolist(),
                             "past_daily_mean": past_mean})
        arrays = [np.asarray(x, dtype=np.float32) for x in zip(*rows)]
        datasets[name] = arrays
        metadata[name] = meta
    qscale = max(float(datasets["train"][6].mean()), 1.0)
    for arrays in datasets.values():
        arrays[5][:, 0] /= qscale
        arrays[6] /= qscale
    config["data_shape"] = list(sales.shape)
    config["split_boundaries_zero_based"] = {"train_end_exclusive": train_end, "validation_end_exclusive": val_end,
                                             "test_end_exclusive": n_days}
    config["scalers_fit_on_training_only"] = {"log1p_demand_max": log_scale, "order_mean": qscale,
                                               "training_daily_mean": train_mean_scale}
    config["sample_counts"] = {k: len(v) for k,v in metadata.items()}
    config["sku_ids"] = [str(x) for x in df.index]
    assert max(m["last_future_day"] for m in metadata["train"]) < min(m["origin_day"] for m in metadata["validation"])
    assert max(m["last_future_day"] for m in metadata["validation"]) < min(m["origin_day"] for m in metadata["test"])
    return {k: TensorDataset(*(torch.from_numpy(a) for a in v)) for k,v in datasets.items()}, metadata


def losses(model, batch, config):
    decision, lead, forecast = model(*batch[:6])
    decision_loss = (decision - batch[6]).square().mean()
    lead_loss = (lead - batch[7]).square().mean()
    error = batch[8].unsqueeze(-1) - forecast
    quantiles = torch.tensor(config["quantiles"], dtype=error.dtype, device=error.device)
    demand_loss = torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
    total = decision_loss + config["lead_loss_weight"] * lead_loss + config["demand_loss_weight"] * demand_loss
    return total, decision_loss, lead_loss, demand_loss


@torch.no_grad()
def evaluate(model, loader, config):
    model.eval()
    totals, count = np.zeros(4), 0
    for batch in loader:
        values = losses(model, batch, config)
        size = len(batch[0])
        totals += size * np.array([v.item() for v in values])
        count += size
    return dict(zip(["total", "decision", "lead", "demand_pinball"], (totals / count).tolist()))


@torch.no_grad()
def predict(model, loader, config):
    model.eval()
    scale = config["scalers_fit_on_training_only"]["order_mean"]
    all_q = []
    for batch in loader:
        all_q.append(model(*batch[:6])[0].cpu().numpy() * scale)
    return np.concatenate(all_q)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4564)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    out = ROOT / "results" / "model_demo"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    config = {"status": "mechanism_demonstration_only_not_original_training_replication",
              "seed": args.seed, "epochs": args.epochs, "history_days": 60, "prediction_days": 21,
              "hidden_size": 50, "context_size": 8, "bridge_size": 8, "quantiles": [.1,.5,.9],
              "lead_layers": [50,20], "integration_layers": [100,100], "dropout": .2,
              "batch_size": 64, "optimizer": "SGD", "learning_rate": .001, "momentum": .85,
              "learning_rate_decay": 1e-4, "weight_initialization": "Gaussian(0,0.01); biases=0",
              "decision_loss": "MSE on normalized theorem-1 order labels; chosen because exact loss was withheld",
              "lead_loss_weight": .01, "demand_loss_weight": .1,
              "b": 9.0, "h": 1.0, "assumed_lead_days": 3, "assumed_review_days": 14,
              "observed_data": "Released five sales series only; no released order/stock table was joined",
              "synthetic_conditions": ["Daily hypothetical order origins (overlapping, correlated examples)",
                                       "Fixed lead time=3 days; fixed review/arrival interval=14 days",
                                       "Starting stock=past28day mean*(lead+review)*deterministic factor in [0.3,1.2]",
                                       "Backlogging convention before arrival: I_arrival=I_order-prearrival demand",
                                       "Inventory cost evaluated only from arrival to next arrival; prearrival cost independent of new order",
                                       "Two calendar covariates are weekly sine/cosine, replacing confidential exogenous features",
                                       "Five SKU one-hot inputs; constant synthetic lead features; no real lead prediction claim"],
              "limits": ["Cannot reproduce the 24,333-SKU 132-feature training experiment",
                         "Not an operational forecast or causal effect estimate",
                         "Scaling and all model selection use training/validation only",
                         "Shared past history is permitted; all future labels stay within their split",
                         "Demand quantile crossing is possible; outputs are not used as calibrated probabilities"],
              "torch_version": torch.__version__, "device": "cpu"}
    label_check = verify_label_formula(args.seed)
    datasets, metadata = build_data(config)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True, generator=generator)
    eval_loaders = {k: DataLoader(v, batch_size=256, shuffle=False) for k,v in datasets.items()}
    model = E2EDemo(config, len(config["sku_ids"]))
    config["parameter_count"] = sum(p.numel() for p in model.parameters())
    write_json(out/"config.json",config)
    print(f"samples={config['sample_counts']} parameters={config['parameter_count']} device=cpu",flush=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1+config["learning_rate_decay"]*epoch))
    initial = {k: evaluate(model, loader, config) for k,loader in eval_loaders.items()}
    initial_parameters = {name: p.detach().clone() for name,p in model.named_parameters()}
    best, best_epoch, history, gradient_check = math.inf, 0, [], None
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = losses(model, batch, config)
            if not all(torch.isfinite(v).item() for v in batch_losses):
                raise FloatingPointError("Nonfinite training loss")
            batch_losses[0].backward()
            if gradient_check is None:
                gradient_check = {name: {"has_gradient": p.grad is not None,
                                         "finite": p.grad is not None and bool(torch.isfinite(p.grad).all()),
                                         "absolute_sum": float(p.grad.abs().sum()) if p.grad is not None else 0.0}
                                  for name,p in model.named_parameters()}
                if not all(x["has_gradient"] and x["finite"] for x in gradient_check.values()):
                    raise AssertionError("Missing or nonfinite parameter gradients")
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        train_result = evaluate(model, eval_loaders["train"], config)
        val_result = evaluate(model, eval_loaders["validation"], config)
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                        **{f"train_{k}":v for k,v in train_result.items()},
                        **{f"validation_{k}":v for k,v in val_result.items()}})
        if val_result["total"] < best:
            best, best_epoch = val_result["total"], epoch
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch, "validation_loss": best, "config": config}, out/"best_validation.pt")
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs} train={train_result['total']:.6f} validation={val_result['total']:.6f}", flush=True)
    elapsed = time.perf_counter()-started
    changed = {name: float((p.detach()-initial_parameters[name]).abs().max()) for name,p in model.named_parameters()}
    saved = torch.load(out/"best_validation.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"])
    final = {k: evaluate(model, loader, config) for k,loader in eval_loaders.items()}
    predictions = predict(model, eval_loaders["test"], config)
    # Check that a freshly constructed model recovers exactly the saved predictions.
    recovered = E2EDemo(config, len(config["sku_ids"]))
    recovered.load_state_dict(saved["model_state"])
    recovery_gap = float(np.max(np.abs(predictions-predict(recovered, eval_loaders["test"], config))))
    rows = []
    for m, prediction in zip(metadata["test"], predictions):
        # Simple past-mean policy, evaluated under the same hypothetical conditions.
        mean_decision = max(m["past_daily_mean"]*(config["assumed_lead_days"]+config["assumed_review_days"])-m["stock_at_order"],0)
        row = {k:v for k,v in m.items() if k != "arrival_demand"}
        row.update({"model_order": float(prediction), "past_mean_order": mean_decision})
        for name,q in [("model",float(prediction)),("past_mean",mean_decision),("oracle",m["oracle_order"])]:
            holding, shortage = interval_cost(q,m["arrival_demand"],m["stock_at_arrival"],config["b"],config["h"])
            row.update({f"{name}_holding":holding,f"{name}_shortage":shortage,f"{name}_total":holding+shortage})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise FloatingPointError("Nonfinite evaluation output")
    oracle_violation = max(float((frame.oracle_total-frame.model_total).max()),
                           float((frame.oracle_total-frame.past_mean_total).max()), 0.0)
    if oracle_violation > 1e-6:
        raise AssertionError("Hindsight oracle failed to provide an interval-cost lower bound")
    frame.to_csv(out/"test_predictions.csv",index=False,encoding="utf-8-sig")
    cost_summary = {name: {kind: float(frame[f"{name}_{kind}"].mean()) for kind in ["holding","shortage","total"]}
                    for name in ["model","past_mean","oracle"]}
    metrics = {"status": config["status"], "best_validation_epoch": best_epoch,"training_seconds":elapsed,
               "initial_losses":initial,"best_checkpoint_losses":final,
               "test_order_MAE":float(np.abs(frame.model_order-frame.oracle_order).mean()),
               "test_order_RMSE":float(np.sqrt(np.square(frame.model_order-frame.oracle_order).mean())),
               "mean_hypothetical_interval_cost":cost_summary,
               "checks":{"theorem_label_bruteforce":label_check,"future_labels_disjoint":True,
                         "checkpoint_max_prediction_difference":recovery_gap,
                         "oracle_lower_bound_max_violation":oracle_violation,
                         "all_gradients_present_and_finite":True,
                         "at_least_one_parameter_updated":any(v>0 for v in changed.values()),
                         "training_total_loss_decreased":final["train"]["total"]<initial["train"]["total"]}}
    if recovery_gap != 0 or not metrics["checks"]["training_total_loss_decreased"]:
        raise AssertionError("Model training/recovery verification failed")
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    write_json(out/"config.json",config)
    write_json(out/"metrics.json",metrics)
    write_json(out/"gradient_check.json",{"first_batch":gradient_check,"maximum_parameter_updates":changed})
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    hist=pd.DataFrame(history)
    axes[0].plot(hist.epoch,hist.train_total,label="Train")
    axes[0].plot(hist.epoch,hist.validation_total,label="Validation")
    axes[0].axvline(best_epoch,linestyle="--",color="gray",label=f"Selected epoch {best_epoch}")
    axes[0].set(xlabel="Epoch",ylabel="Weighted normalized loss",title="E2E mechanism demonstration")
    axes[0].legend(frameon=False)
    x=np.arange(3)
    axes[1].bar(x,[cost_summary[n]["holding"] for n in ["model","past_mean","oracle"]],label="Holding",color="#247BA0")
    axes[1].bar(x,[cost_summary[n]["shortage"] for n in ["model","past_mean","oracle"]],
                bottom=[cost_summary[n]["holding"] for n in ["model","past_mean","oracle"]],label="Shortage",color="#EF8354")
    axes[1].set(xticks=x,xticklabels=["Demo model","Past-mean policy","Hindsight oracle"],ylabel="Mean 14-day scenario cost",
                title="Constructed conditions; not paper results")
    axes[1].legend(frameon=False)
    fig.savefig(out/"learning_curve.png",dpi=170)
    plt.close(fig)
    print(json.dumps({"best_epoch":best_epoch,"test_loss":final["test"]["total"],"costs":cost_summary,
                      "checks":metrics["checks"],"seconds":elapsed},ensure_ascii=True,indent=2),flush=True)


if __name__ == "__main__":
    main()
