# Candidate-latency model regret

The held-out split was evaluated only after the model configuration and uncertainty policy were frozen.

| Model | Split | Exact match | Mean regret | P95 regret | Worst regret |
|---|---|---:|---:|---:|---:|
| candidate_kind_mean | validation | 0.880000 | 0.023365 | 0.005328 | 0.567925 |
| candidate_kind_mean | heldout | 0.920000 | 0.016757 | 0.000000 | 0.418919 |
| analytical | validation | 0.880000 | 0.023365 | 0.005328 | 0.567925 |
| analytical | heldout | 0.840000 | 0.224351 | 0.418919 | 2.847302 |
| ridge | validation | 0.920000 | 0.022930 | 0.000000 | 0.567925 |
| ridge | heldout | 0.920000 | 0.303944 | 0.000000 | 5.129496 |
| single_tree | validation | 0.680000 | 0.869569 | 3.787301 | 7.852000 |
| single_tree | heldout | 0.680000 | 1.253095 | 4.282069 | 6.009009 |
| hybrid | validation | 0.880000 | 0.023365 | 0.005328 | 0.567925 |
| hybrid | heldout | 0.920000 | 0.016757 | 0.000000 | 0.418919 |
| gbdt | validation | 0.880000 | 0.023365 | 0.005328 | 0.567925 |
| gbdt | heldout | 0.920000 | 0.016757 | 0.000000 | 0.418919 |
