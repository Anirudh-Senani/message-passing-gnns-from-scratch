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
- [x] **4.** symmetric_normalize_edge_weights
- [x] **5.** gather_source_node_features
- [x] **6.** scatter_sum_to_nodes
- [x] **7.** scatter_mean_to_nodes
- [x] **8.** scatter_max_to_nodes
- [x] **9.** compute_messages
- [x] **10.** aggregate_messages
- [x] **11.** update_node_features
- [x] **12.** message_passing_layer
- [x] **13.** stack_message_passing_layers
- [x] **14.** gcn_renormalize_adjacency
- [x] **15.** gcn_linear_transform
- [x] **16.** gcn_layer_forward
- [x] **17.** init_gcn_parameters
- [x] **18.** gcn_stack_forward
- [x] **19.** gat_attention_logits
- [x] **20.** gat_masked_neighbor_softmax
- [x] **21.** gat_head_forward
- [x] **22.** merge_gat_heads
- [x] **23.** gat_layer_forward
- [x] **24.** init_gat_parameters
- [x] **25.** gat_stack_forward
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
