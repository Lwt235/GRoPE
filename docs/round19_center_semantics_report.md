# R19 center semantics: accepted exploratory report

2026-10-31. The fixed old-TRAIN screen **passed** and both independent audits passed. The scientific efficacy gate remains **false**, and the research goal remains incomplete. This report reads only accepted outputs; it introduces no new predictions, target subsets, center definitions or parameter selection.

The evidence supports one narrow conclusion: projecting the designated source center retains useful localization information that the projected region mean lost on these three reused development scenes. It permits consideration of a separately frozen Gaussian-parameter hypothesis. It does not demonstrate a successful Gaussian PE.

## Contract and evidence

- Root acceptance: `results/round19_center_root_acceptance_20261031.json`.
- Frozen diagnostic: `ROUND19_CENTER_SEMANTICS_PROTOCOL.md`.
- Accepted producer: `results/round19_center_semantics/summary.json` and `evaluation/rows.json`.
- Independent reconstruction and metrics: `results/round19_center_semantics_independent_audit/{forward_verification,verification,independent_summary,independent_rows}.json`.
- Root verified 69 producer files, 19 auditor files and all nine current/snapshot source identities. Seventeen software-only synthetic tests passed in the target runtime.
- All 48 original TRAIN pairs were retained: 16 each from Harvard c3/c5/c6; q8/q16/q24 with 432 native g8 key cells; 28,224 saved query rows. The exact inherited valid-center counts were 9,521 / 2,463 / 1,096.
- All predictions were globally sealed and independently audited before any target archive was hashed or decoded. Independent source-center reconstruction had maximum absolute discrepancy 1.1368683772161603e-12 px, below the fixed 1e-9 px contract; all nearest indices and flags agreed exactly. Final continuous-metric discrepancy was zero, and discrete counts, Fractions and screen decisions matched exactly.

Here **a** is the R13 projection of the designated source-cell center, using only predicted depth and supplied cameras. **mu** is the accepted projected region mean, converted once from reference-8 units to pixels. **primary** is the fixed R17 h=.25/T=10 argmax. The old means and primary winners are provenance-checked inherited evidence; their original Gaussian integration was not independently rerun. Nearest-a and nearest-mu select the first squared-distance minimum among native key centers, including outside-FOV rows. Neither is the deployed content-aware strong point baseline.

## Frozen screen

PCK uses squared pixel distance <=64 on the original valid-center mask. Each scene contributes one exact correct/valid Fraction; the three scene Fractions have equal weight. Percentages below are displays only; the screen used exact rational arithmetic.

| Query | Nearest-mu PCK (%) | Nearest-a PCK (%) | Fixed R17 primary PCK (%) | a minus mu (pp) | a minus primary (pp) |
|---|---:|---:|---:|---:|---:|
| g8 | 92.830444 | 93.344121 | 93.034772 | 0.513677 | 0.309349 |
| g16 | 92.205357 | 92.831074 | 92.302997 | 0.625717 | 0.528077 |
| g24 | 91.006322 | 93.410081 | 91.935526 | 2.403758 | 1.474554 |

The exact a-minus-mu deltas are:

| Query | Exact delta | Frozen condition | Result |
|---|---:|---|---|
| g8 | 10159795/1977855831 | >= -1/200 | PASS |
| g16 | 2571344/410943645 | >= -1/200 | PASS |
| g24 | 47987/1996332 | >= +1/200 | PASS |

q24 also improved strictly in **each** scene: c3 gained 2/382, c5 gained 14/402, and c6 gained 10/312 correct predictions relative to nearest-mu. Thus all four frozen conditions passed. The q24 macro gain of +2.403758 pp exceeds the +0.5 pp progression threshold. Its +1.474554 pp advantage over the fixed R17 primary is descriptive and was not a second selection condition.

## Complete scene and scale counts

c3/c5/c6 abbreviate `harvard_c3/hv_c3_1`, `harvard_c5/hv_c5_1`, `harvard_c6/hv_c6_1`. Each correct count below uses the displayed original valid count.

| Scene | Query | Saved queries | Valid centers | mu correct | a correct | Primary correct | a minus mu (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| c3 | g8 | 6912 | 3252 | 3128 | 3131 | 3129 | +0.092251 |
| c3 | g16 | 1728 | 851 | 821 | 818 | 821 | -0.352526 |
| c3 | g24 | 768 | 382 | 368 | 370 | 369 | +0.523560 |
| c5 | g8 | 6912 | 3441 | 3348 | 3365 | 3351 | +0.494042 |
| c5 | g16 | 1728 | 882 | 847 | 857 | 852 | +1.133787 |
| c5 | g24 | 768 | 402 | 383 | 397 | 388 | +3.482587 |
| c6 | g8 | 6912 | 2828 | 2404 | 2431 | 2418 | +0.954738 |
| c6 | g16 | 1728 | 730 | 614 | 622 | 612 | +1.095890 |
| c6 | g24 | 768 | 312 | 254 | 264 | 258 | +3.205128 |

For each comparison, **benefit/harm** counts rows where the first prediction becomes correct/incorrect while the second is incorrect/correct. Index disagreement is reported as **all saved rows / original valid rows**. A changed index need not change PCK correctness. No predicted front/FOV mask filters these counts.

| Scene | Query | a vs mu benefit/harm | a vs primary benefit/harm | mu vs primary benefit/harm | a vs mu index disagreement | a vs primary index disagreement | mu vs primary index disagreement |
|---|---|---:|---:|---:|---:|---:|---:|
| c3 | g8 | 8/5 | 11/9 | 7/8 | 164/57 | 1615/207 | 1583/193 |
| c3 | g16 | 4/7 | 3/6 | 2/2 | 81/33 | 354/46 | 320/31 |
| c3 | g24 | 4/2 | 4/3 | 0/1 | 54/23 | 157/27 | 131/11 |
| c5 | g8 | 23/6 | 25/11 | 9/12 | 233/89 | 1613/306 | 1532/262 |
| c5 | g16 | 13/3 | 11/6 | 3/8 | 124/54 | 363/76 | 316/46 |
| c5 | g24 | 14/0 | 9/0 | 2/7 | 72/38 | 178/50 | 144/30 |
| c6 | g8 | 41/14 | 48/35 | 22/36 | 376/140 | 2169/300 | 2040/238 |
| c6 | g16 | 15/7 | 17/7 | 6/4 | 184/80 | 542/105 | 460/59 |
| c6 | g24 | 13/3 | 12/6 | 0/4 | 107/48 | 260/51 | 213/22 |

At q24, a versus mu has 31 benefits and 5 harms across valid rows (net +26 correct); a versus primary has 25 benefits and 9 harms (net +16). These are paired pooled counts, not substitutes for the equal-scene macro used by the screen.

## Continuous distances and counterexamples

Every table cell is **mu -> a**, in pixels. Means, medians and RMSEs use all original valid center distances in the scene/scale; medians are computed from the complete row population, not by averaging pair medians.

| Scene | Query | Mean distance (px) | Median distance (px) | RMSE (px) |
|---|---|---:|---:|---:|
| c3 | g8 | 1.352950 -> 1.342350 | 0.501917 -> 0.495730 | 3.238537 -> 3.322368 |
| c3 | g16 | 1.386598 -> 1.309687 | 0.508732 -> 0.499945 | 3.322020 -> 3.026168 |
| c3 | g24 | 1.300014 -> 1.180157 | 0.510250 -> 0.490707 | 2.841272 -> 2.571439 |
| c5 | g8 | 1.537680 -> 1.447189 | 0.863717 -> 0.825148 | 2.466993 -> 2.311375 |
| c5 | g16 | 1.697403 -> 1.492052 | 0.876602 -> 0.802197 | 2.898349 -> 2.591621 |
| c5 | g24 | 1.883001 -> 1.431560 | 1.044024 -> 0.822001 | 3.022951 -> 2.249295 |
| c6 | g8 | 3.695002 -> 3.615476 | 2.069499 -> 2.069782 | 5.979431 -> 5.832069 |
| c6 | g16 | 3.981416 -> 3.697128 | 2.146891 -> 2.141035 | 6.280348 -> 5.790225 |
| c6 | g24 | 4.147705 -> 3.572428 | 2.278971 -> 2.046297 | 6.718707 -> 5.566848 |

The improvements are not uniform across metrics or populations. In **c3 q8**, mean and median improve and PCK gains three correct rows, but RMSE worsens from **3.238537 to 3.322368 px**. In **c3 q16**, mean/median/RMSE improve, but PCK falls from **821/851 to 818/851** (4 benefits, 7 harms; **-0.352526 pp**). In c6 q8 the median also increases slightly, from 2.069499 to 2.069782 px. These are retained counterexamples; no metric or subset was switched to rescue the screen. The q8/q16 protection condition applies to their equal-scene macros, not to every individual scene.

## Complete correctness transitions

Columns encode `(primary correct, mu correct, a correct)` as three bits. Counts sum to the original valid-center population in each row. This table retains joint wins and losses against the fixed R17 comparator.

| Scene | Query | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c3 | g8 | 110 | 6 | 2 | 5 | 6 | 2 | 3 | 3118 |
| c3 | g16 | 26 | 2 | 1 | 1 | 0 | 2 | 6 | 813 |
| c3 | g24 | 9 | 4 | 0 | 0 | 1 | 0 | 2 | 366 |
| c5 | g8 | 64 | 17 | 1 | 8 | 6 | 6 | 5 | 3334 |
| c5 | g16 | 19 | 8 | 0 | 3 | 3 | 5 | 3 | 841 |
| c5 | g24 | 5 | 7 | 0 | 2 | 0 | 7 | 0 | 381 |
| c6 | g8 | 359 | 29 | 3 | 19 | 24 | 12 | 11 | 2371 |
| c6 | g16 | 100 | 12 | 1 | 5 | 1 | 3 | 6 | 602 |
| c6 | g24 | 42 | 12 | 0 | 0 | 3 | 1 | 3 | 251 |

## Predicted flags and population retention

Counts below are **all saved rows / original valid rows** for the false side of each predicted flag. True counts are the corresponding saved/valid totals above minus these counts. No 3D in-front flag is invented for the reused 2D mean.

| Scene | Query | a behind clamp plane | a outside FOV | mu outside FOV |
|---|---|---:|---:|---:|
| c3 | g8 | 55/0 | 2605/0 | 2607/0 |
| c3 | g16 | 15/0 | 647/0 | 647/0 |
| c3 | g24 | 6/0 | 280/0 | 279/0 |
| c5 | g8 | 0/0 | 2125/4 | 2124/4 |
| c5 | g16 | 0/0 | 517/2 | 517/4 |
| c5 | g24 | 0/0 | 221/1 | 224/3 |
| c6 | g8 | 0/0 | 2707/19 | 2701/18 |
| c6 | g16 | 0/0 | 681/5 | 682/5 |
| c6 | g24 | 0/0 | 297/1 | 293/1 |

Some originally valid target centers have predicted a or mu outside the FOV, and all remain in the PCK and continuous-error calculations. The saved population also retains every predicted-behind row; in this cohort those rows have no valid center target, so they contribute to descriptive all-row flags/index disagreement rather than invented target errors.

## Interpretation and permitted next step

The accepted screen supports testing whether retaining **designated center identity** helps the patch-to-Gaussian construction. It does not show that an attention probability centered at a will preserve these nearest-cell gains. Regional mass, covariance shape, truncation and content competition can still move an argmax or worsen the regional objective. The center a is additional legal input information beyond the old mu/C summary, and must be disclosed as such.

The permitted next step is only a separately frozen test of a compact hypothesis such as mean=a with C_a=C+(mu-a)(mu-a)^T, accompanied by same-a point/fixed controls and all prior strong controls. Its fixed-mean second-moment interpretation is theoretical; no C_a probabilities or new Gaussian candidate were computed in R19. This diagnostic does not establish improved regional KL, six-perturbation robustness, efficiency, learned-V behavior, real optimized-GS behavior or untouched joint confirmation.

In particular, the fixed R17 primary is not the strongest point-localization baseline, and nearest-a is an input-only geometric diagnostic. Beating this one inherited comparator on old TRAIN does not establish that a new PE beats strong point attention. Three repeatedly used development scenes provide no independent generalization evidence. All prior efficacy requirements remain in force.
