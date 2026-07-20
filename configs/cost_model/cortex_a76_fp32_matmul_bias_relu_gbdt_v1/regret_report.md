# Candidate-latency model regret

The held-out split was evaluated only after the model configuration and uncertainty policy were frozen.

| Model | Split | Exact match | Mean regret | P95 regret | Worst regret |
|---|---|---:|---:|---:|---:|
| candidate_kind_mean | validation | 0.777778 | 0.069493 | 0.206522 | 0.418919 |
| candidate_kind_mean | heldout | 0.888889 | 0.009009 | 0.000000 | 0.081081 |
| analytical | validation | 0.777778 | 0.069493 | 0.206522 | 0.418919 |
| analytical | heldout | 0.888889 | 0.009009 | 0.000000 | 0.081081 |
| ridge | validation | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| ridge | heldout | 0.888889 | 0.009009 | 0.000000 | 0.081081 |
| single_tree | validation | 0.888889 | 0.022947 | 0.000000 | 0.206522 |
| single_tree | heldout | 0.888889 | 0.009009 | 0.000000 | 0.081081 |
| hybrid | validation | 0.777778 | 0.069493 | 0.206522 | 0.418919 |
| hybrid | heldout | 0.888889 | 0.009009 | 0.000000 | 0.081081 |
| gbdt | validation | 0.888889 | 0.022947 | 0.000000 | 0.206522 |
| gbdt | heldout | 0.888889 | 0.009009 | 0.000000 | 0.081081 |
