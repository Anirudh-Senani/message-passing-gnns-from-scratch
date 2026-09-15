"""
Message-Passing GNNs from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Scaffold: Message-Passing GNNs (MPNN / GCN / GAT) from scratch in pure PyTorch."""
import numpy as np
import torch


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # Graph primitives on a tiny cycle+chord graph
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    src, dst, n = edges_to_coo(edge_list, num_nodes=4)
    src, dst = add_self_loops(src, dst, n)
    deg = compute_node_degrees(src, dst, n)
    ew = symmetric_normalize_edge_weights(src, dst, n)
    print("degrees:", deg.tolist())
    print("sym-norm weight mean: %.4f" % float(ew.mean()))

    x = torch.arange(n * 6, dtype=torch.float32).view(n, 6) * 0.1
    gathered = gather_source_node_features(x, src)
    summed = scatter_sum_to_nodes(gathered, dst, n)
    print("gather/scatter shapes:", tuple(gathered.shape), tuple(summed.shape))

    def message_fn(h_src, h_dst, edge_attr=None):
        return h_src

    def update_fn(h, agg):
        return torch.relu(h + agg)

    h_mp = message_passing_layer(x, src, dst, message_fn, update_fn, aggr="sum")
    print("MPNN layer out:", tuple(h_mp.shape))

    # GCN / GAT single-layer forwards
    gcn_params = init_gcn_parameters(6, 4, with_bias=True, seed=0)
    h_gcn = gcn_layer_forward(
        x, src, dst,
        gcn_params["weight"], gcn_params.get("bias"),
        num_nodes=n, activation=torch.relu,
    )
    print("GCN out:", tuple(h_gcn.shape))

    heads = init_gat_parameters(6, 4, num_heads=2, with_bias=True, seed=0)
    h_gat, attn = gat_layer_forward(
        x, src, dst, heads, merge_mode="concat", num_nodes=n, activation=torch.relu
    )
    print("GAT (2-head concat) out:", tuple(h_gat.shape), "n_attn_heads:", len(attn))

    # Batched molecule-like graphs + pooling
    graphs = build_graph_regression_dataset(
        4, (6, 10), num_node_features=5, edge_prob=0.35, seed=0
    )
    batch = collate_graph_batch(graphs)
    bx, bb = batch["x"], batch["batch"]
    pooled = global_mean_max_pool(bx, bb)
    print("collated nodes:", int(bx.shape[0]), "pool:", tuple(pooled.shape))

    # Oversmoothing diagnostic on a short GCN stack
    layer_feats = [x]
    h = x
    for seed_i in (1, 2, 3):
        p = init_gcn_parameters(6, 6, seed=seed_i)
        h = gcn_layer_forward(
            h, src, dst, p["weight"], p.get("bias"),
            num_nodes=n, activation=torch.relu,
        )
        layer_feats.append(h.detach())
    os_score = oversmoothing_diagnostic(layer_feats)
    print(
        "oversmoothing mean_similarity: %.4f"
        % float(os_score["mean_similarity"])
    )

    # End-to-end GCN-vs-GAT node classification on a synthetic SBM graph
    result = mpnn_gnn_experiment(
        num_nodes=32,
        num_features=8,
        num_classes=2,
        num_layers=3,
        hidden_dim=16,
        num_epochs=6,
        lr=0.05,
        seed=0,
    )
    print("experiment result type:", type(result).__name__)
    if isinstance(result, dict):
        for key, val in result.items():
            if isinstance(val, float):
                print("  %s: %.4f" % (key, val))
            elif isinstance(val, dict):
                print("  %s:" % key)
                for k2, v2 in val.items():
                    if isinstance(v2, float):
                        print("    %s: %.4f" % (k2, v2))
                    elif isinstance(v2, dict) and "mean_similarity" in v2:
                        print(
                            "    %s.mean_similarity: %.4f"
                            % (k2, float(v2["mean_similarity"]))
                        )
                    elif isinstance(v2, dict) and "loss" in v2:
                        losses = v2.get("loss") or []
                        if losses:
                            print(
                                "    %s.loss final=%.4f (len=%d)"
                                % (k2, float(losses[-1]), len(losses))
                            )
                    else:
                        print("    %s: %s" % (k2, type(v2).__name__))
            else:
                print(
                    "  %s: %r"
                    % (
                        key,
                        val if isinstance(val, (int, str, bool, dict)) else type(val).__name__,
                    )
                )


if __name__ == "__main__":
    main()

