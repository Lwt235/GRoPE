# Final120 versus CE-selected checkpoints: used development data

All15 existing fits and allsix conditions. No new training. This adaptive diagnostic does not rewrite the original negative result or constitute fresh confirmation.

| Method | Selected mCA % | Final120 mCA % | Selected CE | Final120 CE |
|---|---:|---:|---:|---:|
| gaussian_relation | 77.09184 | 77.99329 | 0.70246 | 0.69670 |
| direct_mlp | 76.77573 | 77.17575 | 0.75136 | 0.74566 |
| distance_mlp | 78.42627 | 78.73275 | 0.73671 | 0.73569 |
| center_rope | 71.16275 | 70.93358 | 0.86051 | 0.86215 |
| nope | 69.82864 | 69.46197 | 1.03220 | 1.03283 |

Final Gaussian-minus-control gaps. Intervals resample classes after averaging the three paired seeds.

| Condition | Control | Final gap pp | Paired seeds pp | Class95% pp | Change from selected gap pp |
|---|---|---:|---|---|---:|
| base | direct_mlp | +0.8175 | +2.2510/+2.4452/-2.2436 | [-1.0423876211013894, 2.5816071576035338] | +0.5014 |
| base | distance_mlp | -0.7395 | -0.9960/-0.2830/-0.9394 | [-2.623772913231034, 1.104215188307793] | +0.5950 |
| base | center_rope | +7.0597 | +6.6819/+10.0015/+4.4958 | [4.926602480316249, 9.393963058546394] | +1.1306 |
| base | nope | +8.5313 | +8.2979/+8.7902/+8.5059 | [5.935893799571336, 11.343038464822886] | +1.2681 |
| sparse256 | direct_mlp | +0.7723 | +0.3871/+3.8486/-1.9186 | [-1.6158051099227568, 3.1354453439624996] | +1.1815 |
| sparse256 | distance_mlp | -0.3148 | +0.6679/-0.9004/-0.7118 | [-2.3422510561097503, 1.507517754576579] | +1.6027 |
| sparse256 | center_rope | +6.8765 | +7.9682/+9.5866/+3.0747 | [4.541503882768269, 9.286651033086448] | +1.8787 |
| sparse256 | nope | +9.7593 | +9.3691/+10.4767/+9.4321 | [6.6703814023666945, 12.915282253839905] | +1.9975 |
| dense1024 | direct_mlp | +0.7855 | +1.5932/+1.8633/-1.1002 | [-0.5985901846409087, 2.211412691575736] | +0.8718 |
| dense1024 | distance_mlp | -0.5552 | -0.2323/-0.4028/-1.0306 | [-2.159539080417703, 0.9622031584893902] | +0.4829 |
| dense1024 | center_rope | +5.6695 | +3.2969/+10.2027/+3.5089 | [3.43015408504539, 8.014936105561105] | +0.7464 |
| dense1024 | nope | +8.2551 | +8.0688/+8.6953/+8.0011 | [5.6535669191919204, 10.887018585614602] | +1.0201 |
| jitter02 | direct_mlp | +1.2494 | +2.3212/+2.2551/-0.8282 | [-0.6127953586899664, 3.043913493913495] | +1.2354 |
| jitter02 | distance_mlp | -0.4270 | -0.2186/-0.4708/-0.5915 | [-2.2894848458707164, 1.4625616181322716] | +1.1150 |
| jitter02 | center_rope | +7.3674 | +7.4430/+10.4449/+4.2143 | [5.265423673016383, 9.576945213563395] | +2.0669 |
| jitter02 | nope | +8.9743 | +8.4793/+9.2346/+9.2089 | [6.427260258883233, 11.709683435663827] | +1.8131 |
| tilt | direct_mlp | +0.4952 | +2.9231/+1.1648/-2.6022 | [-1.6800579104793727, 2.422504052675088] | -0.2464 |
| tilt | distance_mlp | -0.7894 | -0.5931/-0.8581/-0.9170 | [-2.736958874458874, 1.1053419540738385] | +0.6838 |
| tilt | center_rope | +6.5992 | +5.7901/+8.7398/+5.2678 | [4.241290443536287, 9.10194601889774] | +1.0221 |
| tilt | nope | +8.7130 | +9.4036/+7.8374/+8.8981 | [6.269313672635816, 11.371957915430846] | +0.7059 |
| permuted | direct_mlp | +0.8175 | +2.2510/+2.4452/-2.2436 | [-1.0423876211013894, 2.5816071576035338] | +0.5014 |
| permuted | distance_mlp | -0.7395 | -0.9960/-0.2830/-0.9394 | [-2.623772913231034, 1.104215188307793] | +0.5950 |
| permuted | center_rope | +7.0597 | +6.6819/+10.0015/+4.4958 | [4.926602480316249, 9.393963058546394] | +1.1306 |
| permuted | nope | +8.5313 | +8.2979/+8.7902/+8.5059 | [5.935893799571336, 11.343038464822886] | +1.2681 |

Final base descriptive criterion: False. Scientific and actual image review complete; see review.json. Full role changes, source metrics, seed differences and gap interactions are in result.json.

The mean Gaussian improvement is insufficient to establish a strong-control lead. Both the original selected-policy result and this final120 result remain negative for that claim. End the fixed checkpoint diagnostic; further work concerns the PE design, not checkpoint search.

The left plot uses blue dots for paired seeds and black diamonds for means. Right intervals resample classes after averaging paired seeds.

![Checkpoint policy comparison](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/checkpoint_policy1/checkpoint_policy.png)
