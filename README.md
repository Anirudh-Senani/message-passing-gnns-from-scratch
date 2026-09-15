# Message-Passing GNNs from Scratch

Implement graph primitives, the Gilmer MPNN triad, Kipf–Welling GCN, and multi-head GAT entirely in pure PyTorch—no torch_geometric. Build synthetic SBM and molecule-like datasets, training loops, an oversmoothing diagnostic, and an end-to-end GCN-vs-GAT experiment to understand how modern GNNs really work.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** edges_to_coo
- [x] **2.** add_self_loops
- [x] **3.** compute_node_degrees
- [ ] **4.** symmetric_normalize_edge_weights
- [ ] **5.** gather_source_node_features
- [ ] **6.** scatter_sum_to_nodes
- [ ] **7.** scatter_mean_to_nodes
- [ ] **8.** scatter_max_to_nodes
- [ ] **9.** compute_messages
- [ ] **10.** aggregate_messages
- [ ] **11.** update_node_features
- [ ] **12.** message_passing_layer
- [ ] **13.** stack_message_passing_layers
- [ ] **14.** gcn_renormalize_adjacency
- [ ] **15.** gcn_linear_transform
- [ ] **16.** gcn_layer_forward
- [ ] **17.** init_gcn_parameters
- [ ] **18.** gcn_stack_forward
- [ ] **19.** gat_attention_logits
- [ ] **20.** gat_masked_neighbor_softmax
- [ ] **21.** gat_head_forward
- [ ] **22.** merge_gat_heads
- [ ] **23.** gat_layer_forward
- [ ] **24.** init_gat_parameters
- [ ] **25.** gat_stack_forward
- [ ] **26.** global_mean_pool
- [ ] **27.** global_sum_pool
- [ ] **28.** global_max_pool
- [ ] **29.** global_mean_max_pool
- [ ] **30.** node_classification_head
- [ ] **31.** graph_regression_head
- [ ] **32.** generate_sbm_graph
- [ ] **33.** build_node_classification_dataset
- [ ] **34.** generate_molecule_like_graph
- [ ] **35.** build_graph_regression_dataset
- [ ] **36.** collate_graph_batch
- [ ] **37.** cross_entropy_loss
- [ ] **38.** mse_loss
- [ ] **39.** accuracy_metric
- [ ] **40.** mae_metric
- [ ] **41.** gnn_train_step
- [ ] **42.** train_node_classifier
- [ ] **43.** train_graph_regressor
- [ ] **44.** representation_similarity
- [ ] **45.** oversmoothing_diagnostic
- [ ] **46.** mpnn_gnn_experiment

---

Built on Deep-ML.
