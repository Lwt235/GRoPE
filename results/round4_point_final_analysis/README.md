# Round 4 point verification

All70 models completed full dataset, validation and eight-condition checkpoint replay before initial bootstrap integer-output casting failure. This finalizer independently recomputes primary metrics and float64 paired bootstrap from350 saved probability archives; no retraining or checkpoint reselection.

Validation-selected Gaussian scales: {'oracle_train': 'gaussian200_full', 'robust_train': 'gaussian200_full'}. Maximum independent metric error: 1.03e-07.

| Training | Condition | Method | Visible R1 | Closed R1 | NLL | ECE | FMR |
|---|---|---|---:|---:|---:|---:|---:|
| oracle_train | oracle | projected_rope | 0.966719 | 0.979402 | 0.461012 | 0.049769 | 0.779839 |
| oracle_train | oracle | gaussian_full | 0.973440 | 0.982084 | 0.440815 | 0.063884 | 0.727436 |
| oracle_train | oracle | gaussian_none | 0.970144 | 0.980379 | 0.455019 | 0.059544 | 0.756950 |
| oracle_train | oracle | gaussian25_full | 0.969292 | 0.980121 | 0.452818 | 0.053555 | 0.763648 |
| oracle_train | oracle | gaussian50_full | 0.971073 | 0.981727 | 0.448895 | 0.057933 | 0.753469 |
| oracle_train | oracle | gaussian200_full | 0.976812 | 0.984157 | 0.428486 | 0.071233 | 0.695919 |
| oracle_train | oracle | resolvent_full | 0.969849 | 0.980901 | 0.447652 | 0.056068 | 0.749728 |
| oracle_train | mixed | projected_rope | 0.965533 | 0.978527 | 0.462256 | 0.049196 | 0.778187 |
| oracle_train | mixed | gaussian_full | 0.962536 | 0.975157 | 0.457053 | 0.063625 | 0.722957 |
| oracle_train | mixed | gaussian_none | 0.969369 | 0.980324 | 0.456239 | 0.058623 | 0.757036 |
| oracle_train | mixed | gaussian25_full | 0.966653 | 0.978000 | 0.456539 | 0.052490 | 0.767603 |
| oracle_train | mixed | gaussian50_full | 0.966261 | 0.977678 | 0.456550 | 0.056693 | 0.755491 |
| oracle_train | mixed | gaussian200_full | 0.952424 | 0.970239 | 0.466104 | 0.073377 | 0.675121 |
| oracle_train | mixed | resolvent_full | 0.965761 | 0.977631 | 0.453310 | 0.054633 | 0.751099 |
| oracle_train | scale_noise | projected_rope | 0.966719 | 0.979402 | 0.461012 | 0.049769 | 0.779839 |
| oracle_train | scale_noise | gaussian_full | 0.936893 | 0.959703 | 0.497549 | 0.067184 | 0.715935 |
| oracle_train | scale_noise | gaussian_none | 0.970144 | 0.980379 | 0.455019 | 0.059544 | 0.756950 |
| oracle_train | scale_noise | gaussian25_full | 0.963594 | 0.974165 | 0.461191 | 0.051462 | 0.771116 |
| oracle_train | scale_noise | gaussian50_full | 0.956383 | 0.968739 | 0.471694 | 0.056782 | 0.757901 |
| oracle_train | scale_noise | gaussian200_full | 0.891202 | 0.938002 | 0.569760 | 0.088835 | 0.615205 |
| oracle_train | scale_noise | resolvent_full | 0.956763 | 0.970450 | 0.464496 | 0.053418 | 0.754433 |
| oracle_train | rgb_pred | projected_rope | 0.933758 | 0.946136 | 0.633242 | 0.048219 | 0.819199 |
| oracle_train | rgb_pred | gaussian_full | 0.936929 | 0.945618 | 0.614899 | 0.056387 | 0.756618 |
| oracle_train | rgb_pred | gaussian_none | 0.936882 | 0.946641 | 0.617891 | 0.046762 | 0.792035 |
| oracle_train | rgb_pred | gaussian25_full | 0.935770 | 0.945305 | 0.621723 | 0.047607 | 0.794996 |
| oracle_train | rgb_pred | gaussian50_full | 0.937601 | 0.946395 | 0.618580 | 0.050867 | 0.779657 |
| oracle_train | rgb_pred | gaussian200_full | 0.940474 | 0.945606 | 0.600647 | 0.059835 | 0.710860 |
| oracle_train | rgb_pred | resolvent_full | 0.936520 | 0.945773 | 0.616349 | 0.050140 | 0.777515 |
| robust_train | oracle | projected_rope | 0.966450 | 0.980692 | 0.463783 | 0.059014 | 0.773425 |
| robust_train | oracle | gaussian_full | 0.971762 | 0.981784 | 0.440569 | 0.062639 | 0.727236 |
| robust_train | oracle | gaussian_none | 0.970644 | 0.979345 | 0.458211 | 0.064331 | 0.762174 |
| robust_train | oracle | gaussian25_full | 0.967982 | 0.979647 | 0.455957 | 0.057991 | 0.759901 |
| robust_train | oracle | gaussian50_full | 0.965531 | 0.980431 | 0.450065 | 0.062115 | 0.732652 |
| robust_train | oracle | gaussian200_full | 0.978535 | 0.984164 | 0.430769 | 0.062712 | 0.723743 |
| robust_train | oracle | resolvent_full | 0.970825 | 0.980276 | 0.447863 | 0.058175 | 0.748940 |
| robust_train | mixed | projected_rope | 0.965056 | 0.979563 | 0.465033 | 0.058020 | 0.773487 |
| robust_train | mixed | gaussian_full | 0.960908 | 0.974386 | 0.456904 | 0.063088 | 0.719760 |
| robust_train | mixed | gaussian_none | 0.969700 | 0.979081 | 0.459340 | 0.063538 | 0.762811 |
| robust_train | mixed | gaussian25_full | 0.965140 | 0.977476 | 0.459680 | 0.056432 | 0.760976 |
| robust_train | mixed | gaussian50_full | 0.961348 | 0.977218 | 0.457694 | 0.061938 | 0.731137 |
| robust_train | mixed | gaussian200_full | 0.955981 | 0.971024 | 0.465721 | 0.065530 | 0.701764 |
| robust_train | mixed | resolvent_full | 0.966844 | 0.977528 | 0.453366 | 0.057464 | 0.748027 |
| robust_train | scale_noise | projected_rope | 0.966450 | 0.980692 | 0.463783 | 0.059014 | 0.773425 |
| robust_train | scale_noise | gaussian_full | 0.937663 | 0.959042 | 0.497140 | 0.070310 | 0.708438 |
| robust_train | scale_noise | gaussian_none | 0.970644 | 0.979345 | 0.458211 | 0.064331 | 0.762174 |
| robust_train | scale_noise | gaussian25_full | 0.961527 | 0.973857 | 0.464325 | 0.055307 | 0.769220 |
| robust_train | scale_noise | gaussian50_full | 0.952102 | 0.968851 | 0.472728 | 0.062573 | 0.738277 |
| robust_train | scale_noise | gaussian200_full | 0.902552 | 0.939860 | 0.560742 | 0.084558 | 0.650002 |
| robust_train | scale_noise | resolvent_full | 0.957132 | 0.970143 | 0.464126 | 0.056216 | 0.749887 |
| robust_train | rgb_pred | projected_rope | 0.932364 | 0.945980 | 0.632680 | 0.050642 | 0.807111 |
| robust_train | rgb_pred | gaussian_full | 0.938430 | 0.945722 | 0.602122 | 0.050187 | 0.751308 |
| robust_train | rgb_pred | gaussian_none | 0.937204 | 0.945559 | 0.611079 | 0.047426 | 0.792644 |
| robust_train | rgb_pred | gaussian25_full | 0.934879 | 0.945043 | 0.618157 | 0.047186 | 0.784305 |
| robust_train | rgb_pred | gaussian50_full | 0.932995 | 0.945513 | 0.614758 | 0.050583 | 0.755812 |
| robust_train | rgb_pred | gaussian200_full | 0.942095 | 0.945668 | 0.597039 | 0.051394 | 0.732962 |
| robust_train | rgb_pred | resolvent_full | 0.937251 | 0.945616 | 0.612787 | 0.049582 | 0.778270 |

The Gaussian quarter/half/full/double scale grid is reported in full. Point queries use known fixed covariance; the point experiment has no resolvent_none control, so its support contribution is established separately by the symmetric real-region ablations.
