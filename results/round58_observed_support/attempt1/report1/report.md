# R58 observed support relations: development results

All 18 fits and 72 condition groups completed. Scientific interpretation and visual review are still required.
This is an old-ModelNet local-correspondence experiment, not fresh confirmation, optimized 3DGS, or full registration.
R57 remains not promoted. The complete research goal remains unachieved.

## base

| Method | Region R@1 % | Center PCK % | Selected overlap | Hard EPE | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian relation | 34.458 | 19.374 | 0.231366 | 0.377800 | 0.331958 | 3.465510 |
| Gaussian metric only | 34.274 | 19.221 | 0.230101 | 0.379655 | 0.331812 | 3.470909 |
| Euclidean distance | 35.135 | 19.970 | 0.234040 | 0.375103 | 0.330620 | 3.450592 |
| Distance + angle | 34.951 | 19.808 | 0.232976 | 0.377287 | 0.332029 | 3.455356 |
| Invariant MLP | 38.499 | 22.504 | 0.253717 | 0.357833 | 0.316488 | 3.352861 |
| No explicit PE | 17.683 | 8.789 | 0.124559 | 0.512965 | 0.416250 | 3.940711 |

Primary minus controls (paired seeds11/22/33; pp for recall/PCK):

| Control | Region differences pp | Center differences pp | Soft EPE differences |
| --- | --- | --- | --- |
| Gaussian metric only | +0.9333/-0.1658/-0.2168 | +0.7026/-0.1479/-0.0956 | -0.0051/+0.0025/+0.0030 |
| Euclidean distance | -1.0315/-0.6592/-0.3404 | -0.5980/-0.5266/-0.6618 | +0.0013/+0.0011/+0.0016 |
| Distance + angle | -0.9741/-0.4399/-0.0638 | -0.7204/-0.1581/-0.4220 | -0.0013/+0.0017/-0.0006 |
| Invariant MLP | -4.7623/-3.9259/-3.4350 | -3.6020/-2.8293/-2.9569 | +0.0152/+0.0156/+0.0157 |
| No explicit PE | +16.4061/+16.9417/+16.9774 | +10.2476/+10.8112/+10.6990 | -0.0857/-0.0838/-0.0834 |

Eligible query counts: [39245, 39183]; fraction 0.634942. Nearest-center oracle PCK 82.340%; best overlap mean 0.603324.
The saved best-overlap center oracle uses the first maximizing index; ties can affect it. The separate validation tie diagnostic measures this explicitly.

## noise02

| Method | Region R@1 % | Center PCK % | Selected overlap | Hard EPE | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian relation | 32.343 | 16.643 | 0.215411 | 0.393838 | 0.344109 | 3.545268 |
| Gaussian metric only | 32.178 | 16.508 | 0.214302 | 0.394300 | 0.343282 | 3.547612 |
| Euclidean distance | 33.023 | 17.146 | 0.218389 | 0.390750 | 0.342582 | 3.525380 |
| Distance + angle | 32.967 | 17.061 | 0.217992 | 0.392714 | 0.343603 | 3.525427 |
| Invariant MLP | 36.546 | 19.813 | 0.238382 | 0.372575 | 0.328397 | 3.425747 |
| No explicit PE | 16.018 | 7.291 | 0.113023 | 0.530138 | 0.425815 | 3.999441 |

Primary minus controls (paired seeds11/22/33; pp for recall/PCK):

| Control | Region differences pp | Center differences pp | Soft EPE differences |
| --- | --- | --- | --- |
| Gaussian metric only | +0.8007/-0.0347/-0.2725 | +0.5437/+0.0655/-0.2056 | -0.0050/+0.0033/+0.0042 |
| Euclidean distance | -0.6709/-0.6311/-0.7390 | -0.3239/-0.3946/-0.7917 | +0.0001/+0.0028/+0.0017 |
| Distance + angle | -0.8316/-0.7455/-0.2969 | -0.4023/-0.3534/-0.5000 | -0.0002/+0.0026/-0.0009 |
| Invariant MLP | -5.0832/-4.5241/-3.0037 | -3.7003/-3.1952/-2.6142 | +0.0173/+0.0187/+0.0112 |
| No explicit PE | +16.3948/+15.9424/+16.6365 | +9.4056/+9.1395/+9.5097 | -0.0845/-0.0794/-0.0812 |

Eligible query counts: [38897, 38908]; fraction 0.629898. Nearest-center oracle PCK 81.469%; best overlap mean 0.593332.
The saved best-overlap center oracle uses the first maximizing index; ties can affect it. The separate validation tie diagnostic measures this explicitly.

## sparse256

| Method | Region R@1 % | Center PCK % | Selected overlap | Hard EPE | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian relation | 35.065 | 19.211 | 0.229294 | 0.349106 | 0.307840 | 3.805903 |
| Gaussian metric only | 34.936 | 19.240 | 0.228821 | 0.351321 | 0.308064 | 3.816034 |
| Euclidean distance | 37.226 | 21.266 | 0.239858 | 0.337022 | 0.299212 | 3.783111 |
| Distance + angle | 37.605 | 21.372 | 0.241386 | 0.337665 | 0.299600 | 3.788902 |
| Invariant MLP | 39.140 | 22.538 | 0.249387 | 0.330735 | 0.294675 | 3.714421 |
| No explicit PE | 23.507 | 12.062 | 0.161441 | 0.440920 | 0.368418 | 4.021318 |

Primary minus controls (paired seeds11/22/33; pp for recall/PCK):

| Control | Region differences pp | Center differences pp | Soft EPE differences |
| --- | --- | --- | --- |
| Gaussian metric only | +0.6766/-0.2409/-0.0496 | +0.4836/-0.3826/-0.1860 | -0.0043/+0.0019/+0.0017 |
| Euclidean distance | -2.6251/-2.7774/-1.0805 | -2.4107/-2.1840/-1.5676 | +0.0079/+0.0134/+0.0046 |
| Distance + angle | -3.2326/-2.6233/-1.7624 | -2.5383/-1.8439/-2.0990 | +0.0072/+0.0115/+0.0060 |
| Invariant MLP | -5.0429/-4.8108/-2.3700 | -4.1926/-3.6099/-2.1769 | +0.0145/+0.0175/+0.0075 |
| No explicit PE | +11.2087/+11.2778/+12.1865 | +6.7823/+6.9240/+7.7405 | -0.0616/-0.0573/-0.0629 |

Eligible query counts: [28250, 28206]; fraction 0.457060. Nearest-center oracle PCK 86.531%; best overlap mean 0.529999.
The saved best-overlap center oracle uses the first maximizing index; ties can affect it. The separate validation tie diagnostic measures this explicitly.

## low_overlap

| Method | Region R@1 % | Center PCK % | Selected overlap | Hard EPE | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian relation | 33.052 | 18.138 | 0.227709 | 0.369273 | 0.325583 | 3.643536 |
| Gaussian metric only | 33.295 | 18.393 | 0.229401 | 0.366817 | 0.323302 | 3.629579 |
| Euclidean distance | 32.688 | 18.063 | 0.225243 | 0.369591 | 0.325584 | 3.655510 |
| Distance + angle | 32.489 | 17.786 | 0.223230 | 0.371783 | 0.326371 | 3.657201 |
| Invariant MLP | 35.692 | 19.957 | 0.243541 | 0.355704 | 0.315678 | 3.588736 |
| No explicit PE | 18.961 | 9.408 | 0.135099 | 0.473043 | 0.394321 | 3.962435 |

Primary minus controls (paired seeds11/22/33; pp for recall/PCK):

| Control | Region differences pp | Center differences pp | Soft EPE differences |
| --- | --- | --- | --- |
| Gaussian metric only | -0.1999/-0.3631/-0.1649 | -0.3747/-0.3981/+0.0067 | +0.0012/+0.0019/+0.0037 |
| Euclidean distance | +0.9327/+0.1549/+0.0067 | +0.1682/+0.4414/-0.3847 | -0.0054/+0.0022/+0.0032 |
| Distance + angle | +0.5946/+1.0659/+0.0283 | +0.2465/+0.9127/-0.1049 | -0.0042/+0.0002/+0.0016 |
| Invariant MLP | -3.6508/-2.0019/-2.2651 | -2.8797/-1.0676/-1.5089 | +0.0079/+0.0094/+0.0124 |
| No explicit PE | +13.8786/+13.8103/+14.5848 | +8.2692/+8.8538/+9.0670 | -0.0715/-0.0673/-0.0675 |

Eligible query counts: [30131, 29911]; fraction 0.486091. Nearest-center oracle PCK 84.108%; best overlap mean 0.653664.
The saved best-overlap center oracle uses the first maximizing index; ties can affect it. The separate validation tie diagnostic measures this explicitly.

## Optimization and checks

| Method | PE parameters | Selected epochs | Cumulative fit seconds |
| --- | ---: | --- | ---: |
| Gaussian relation | 9216 | 45/39/33 | 1454.40 |
| Gaussian metric only | 9216 | 45/36/39 | 1426.61 |
| Euclidean distance | 9216 | 42/39/39 | 1393.30 |
| Distance + angle | 18432 | 39/45/33 | 2933.21 |
| Invariant MLP | 10368 | 45/57/45 | 1501.00 |
| No explicit PE | 0 | 48/54/51 | 1369.48 |

Total fitting/selection time: 10077.99s. Shared-host timings are not a controlled speed benchmark.
Max independent metric error: 8.88e-16. Actual-input direct Gaussian scalar error: 3.55e-15.
Validation selection, source/cache identity, shared initialization and actual batch order were checked.
Independent arithmetic here means a separate NumPy path by the same author, not independent laboratory replication.

Object-bootstrap intervals in comparisons.json are conditional on the three fitted models. They do not estimate training-seed uncertainty or make reused objects fresh confirmation. No multiplicity-adjusted hypothesis-test claim is made.

## Input repeatability diagnostic

On old validation data, 107710 fixed sampled relations were compared. 144 of 1930 directions had no eligible relation.

| Relation | Mean symmetric relative change | Mean sine-input RMSE |
| --- | ---: | ---: |
| Euclidean distance | 0.094429 | 0.037324 |
| Gaussian metric only | 0.179075 | 0.207318 |
| Gaussian relation | 0.174191 | 0.207944 |

This measures combined sampling/center/support sensitivity. It does not isolate covariance estimation error, predict task performance, or create an additional acceptance gate. The sine RMSE also depends on the fixed argument scales. The full Gaussian relation is slightly less variable than its metric-only control under relative error; the trace term is not uniquely implicated.
[Diagnostic plan](/home/liwt/GRoPE/docs/round58_development_diagnostics.md); [interpretation](/home/liwt/GRoPE/results/round58_observed_support/attempt1/diagnostics/repeatability/interpretation.md).

## Interpretation still required

Assess all paired strong-control effects, region/center agreement and robustness before any next decision. A local pass cannot establish general effectiveness or erase previous classification failures.

![Paired comparisons](/home/liwt/GRoPE/results/round58_observed_support/attempt1/report1/comparison.png)

![Learning curves](/home/liwt/GRoPE/results/round58_observed_support/attempt1/report1/learning_curves.png)
