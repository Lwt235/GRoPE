# Compact Gaussian PE: point/rejection negative controls

All 50 fixed runs completed. These are repeats on previously examined synthetic namespaces, not new-data discovery evidence. The soft real-region task and this hard point-label task have different target semantics. Ordinary gradients must remain finite; no post-failure tuning is allowed.

## Absolute means

| Training | Condition | Method | Visible R1 | Closed R1 | NLL | Unmatched AP | ECE | FMR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| oracle_train | oracle | projected_rope | 0.9646 | 0.9773 | 0.4958 | 0.5613 | 0.0521 | 0.7798 |
| oracle_train | oracle | expected_fixed | 0.9664 | 0.9834 | 0.5337 | 0.4996 | 0.0767 | 0.8498 |
| oracle_train | oracle | expected_shape | 0.9616 | 0.9823 | 0.5548 | 0.4518 | 0.0701 | 0.8583 |
| oracle_train | oracle | normalized_fixed | 0.9681 | 0.9788 | 0.4871 | 0.5878 | 0.0561 | 0.7672 |
| oracle_train | oracle | normalized_shape | 0.9700 | 0.9799 | 0.4728 | 0.6302 | 0.0620 | 0.7390 |
| oracle_train | mixed | projected_rope | 0.9618 | 0.9759 | 0.4972 | 0.5580 | 0.0503 | 0.7769 |
| oracle_train | mixed | expected_fixed | 0.9641 | 0.9811 | 0.5351 | 0.4961 | 0.0756 | 0.8480 |
| oracle_train | mixed | expected_shape | 0.9066 | 0.9505 | 0.6244 | 0.3644 | 0.0474 | 0.8173 |
| oracle_train | mixed | normalized_fixed | 0.9661 | 0.9772 | 0.4885 | 0.5845 | 0.0545 | 0.7656 |
| oracle_train | mixed | normalized_shape | 0.9555 | 0.9696 | 0.4904 | 0.6047 | 0.0602 | 0.7333 |
| oracle_train | scale_noise | projected_rope | 0.9646 | 0.9773 | 0.4958 | 0.5613 | 0.0521 | 0.7798 |
| oracle_train | scale_noise | expected_fixed | 0.9664 | 0.9834 | 0.5337 | 0.4996 | 0.0767 | 0.8498 |
| oracle_train | scale_noise | expected_shape | 0.7635 | 0.8800 | 0.8088 | 0.3074 | 0.0241 | 0.6881 |
| oracle_train | scale_noise | normalized_fixed | 0.9681 | 0.9788 | 0.4871 | 0.5878 | 0.0561 | 0.7672 |
| oracle_train | scale_noise | normalized_shape | 0.9286 | 0.9531 | 0.5332 | 0.5654 | 0.0643 | 0.7250 |
| oracle_train | rgb_pred | projected_rope | 0.9234 | 0.9367 | 0.6949 | 0.4702 | 0.0508 | 0.8227 |
| oracle_train | rgb_pred | expected_fixed | 0.9233 | 0.9410 | 0.7405 | 0.3997 | 0.0804 | 0.8882 |
| oracle_train | rgb_pred | expected_shape | 0.9191 | 0.9403 | 0.7549 | 0.3554 | 0.0773 | 0.9151 |
| oracle_train | rgb_pred | normalized_fixed | 0.9275 | 0.9377 | 0.6795 | 0.5124 | 0.0478 | 0.8021 |
| oracle_train | rgb_pred | normalized_shape | 0.9279 | 0.9360 | 0.6724 | 0.5479 | 0.0521 | 0.7666 |
| robust_train | oracle | projected_rope | 0.9616 | 0.9774 | 0.4964 | 0.5671 | 0.0660 | 0.7605 |
| robust_train | oracle | expected_fixed | 0.9675 | 0.9834 | 0.5384 | 0.5006 | 0.0768 | 0.8575 |
| robust_train | oracle | expected_shape | 0.9741 | 0.9824 | 0.5558 | 0.4632 | 0.0706 | 0.8842 |
| robust_train | oracle | normalized_fixed | 0.9675 | 0.9774 | 0.4903 | 0.5877 | 0.0570 | 0.7681 |
| robust_train | oracle | normalized_shape | 0.9706 | 0.9794 | 0.4725 | 0.6350 | 0.0607 | 0.7414 |
| robust_train | mixed | projected_rope | 0.9586 | 0.9756 | 0.4977 | 0.5642 | 0.0635 | 0.7607 |
| robust_train | mixed | expected_fixed | 0.9653 | 0.9813 | 0.5397 | 0.4972 | 0.0758 | 0.8554 |
| robust_train | mixed | expected_shape | 0.9223 | 0.9508 | 0.6201 | 0.3733 | 0.0514 | 0.8455 |
| robust_train | mixed | normalized_fixed | 0.9659 | 0.9764 | 0.4916 | 0.5843 | 0.0550 | 0.7663 |
| robust_train | mixed | normalized_shape | 0.9573 | 0.9700 | 0.4894 | 0.6113 | 0.0595 | 0.7350 |
| robust_train | scale_noise | projected_rope | 0.9616 | 0.9774 | 0.4964 | 0.5671 | 0.0660 | 0.7605 |
| robust_train | scale_noise | expected_fixed | 0.9675 | 0.9834 | 0.5384 | 0.5006 | 0.0768 | 0.8575 |
| robust_train | scale_noise | expected_shape | 0.7864 | 0.8817 | 0.7890 | 0.3148 | 0.0255 | 0.7225 |
| robust_train | scale_noise | normalized_fixed | 0.9675 | 0.9774 | 0.4903 | 0.5877 | 0.0570 | 0.7681 |
| robust_train | scale_noise | normalized_shape | 0.9315 | 0.9534 | 0.5313 | 0.5723 | 0.0659 | 0.7217 |
| robust_train | rgb_pred | projected_rope | 0.9195 | 0.9364 | 0.6814 | 0.4795 | 0.0502 | 0.8025 |
| robust_train | rgb_pred | expected_fixed | 0.9249 | 0.9411 | 0.7409 | 0.4004 | 0.0774 | 0.8987 |
| robust_train | rgb_pred | expected_shape | 0.9300 | 0.9401 | 0.7507 | 0.3677 | 0.0793 | 0.9441 |
| robust_train | rgb_pred | normalized_fixed | 0.9280 | 0.9372 | 0.6731 | 0.5139 | 0.0446 | 0.8025 |
| robust_train | rgb_pred | normalized_shape | 0.9292 | 0.9361 | 0.6689 | 0.5530 | 0.0501 | 0.7655 |

## Paired differences

| Training | Condition | Control | Full-minus-control visible R1 | 95% seed-scene CI |
|---|---|---|---:|---|
| oracle_train | oracle | normalized_fixed | +0.001960 | [9.798942105739656e-06, 0.0038640173508497537] |
| oracle_train | oracle | expected_shape | +0.008460 | [-0.005881451327231027, 0.027508950435205345] |
| oracle_train | oracle | projected_rope | +0.005453 | [0.0010378031264096908, 0.010011541183594366] |
| oracle_train | oracle | expected_fixed | +0.003631 | [-0.005355411919543484, 0.012379712701102519] |
| oracle_train | mixed | normalized_fixed | -0.010584 | [-0.012986956347861138, -0.008277724980501914] |
| oracle_train | mixed | expected_shape | +0.048904 | [0.03261595288062798, 0.06640525923032616] |
| oracle_train | mixed | projected_rope | -0.006306 | [-0.010997114327547146, -0.0014048148863054922] |
| oracle_train | mixed | expected_fixed | -0.008545 | [-0.017888437096487395, -0.0002499056282571907] |
| oracle_train | depth_fit | normalized_fixed | +0.004549 | [0.0014458164700916968, 0.007465484894499217] |
| oracle_train | depth_fit | expected_shape | +0.024638 | [0.010870168697327702, 0.04365824272435753] |
| oracle_train | depth_fit | projected_rope | +0.007032 | [0.0026660064574646294, 0.011112298739672738] |
| oracle_train | depth_fit | expected_fixed | +0.008115 | [-0.0007719850109092797, 0.01732991253844959] |
| oracle_train | scale_noise | normalized_fixed | -0.039480 | [-0.046347783126575816, -0.0330071533899923] |
| oracle_train | scale_noise | expected_shape | +0.165057 | [0.14460505804678192, 0.18631877012310424] |
| oracle_train | scale_noise | projected_rope | -0.035987 | [-0.04539268631200742, -0.02673553499360617] |
| oracle_train | scale_noise | expected_fixed | -0.037809 | [-0.05110372160506523, -0.02790260711886518] |
| oracle_train | rgb_pred | normalized_fixed | +0.000378 | [-0.002476783016101003, 0.0028599855875137458] |
| oracle_train | rgb_pred | expected_shape | +0.008729 | [-0.005420712842283456, 0.0267187203592027] |
| oracle_train | rgb_pred | projected_rope | +0.004458 | [-0.0005894155201043133, 0.009833422174332094] |
| oracle_train | rgb_pred | expected_fixed | +0.004540 | [-0.006311703031363427, 0.014705059943487501] |
| robust_train | oracle | normalized_fixed | +0.003112 | [0.0014754006947681397, 0.004759080355040303] |
| robust_train | oracle | expected_shape | -0.003489 | [-0.010662194446690091, 0.004617960606955898] |
| robust_train | oracle | projected_rope | +0.008974 | [0.005527948882378755, 0.012898606938191315] |
| robust_train | oracle | expected_fixed | +0.003059 | [-0.009738812626265742, 0.01745987280286431] |
| robust_train | mixed | normalized_fixed | -0.008639 | [-0.011861493868542123, -0.005333691634472828] |
| robust_train | mixed | expected_shape | +0.035019 | [0.024029773051173785, 0.04632370528831322] |
| robust_train | mixed | projected_rope | -0.001303 | [-0.005708384972305654, 0.0030719313730932047] |
| robust_train | mixed | expected_fixed | -0.008059 | [-0.020526195544672835, 0.005028920234772322] |
| robust_train | depth_fit | normalized_fixed | +0.005274 | [0.003268678684518812, 0.007556898840338745] |
| robust_train | depth_fit | expected_shape | +0.011945 | [0.004296403982154094, 0.02103709345410954] |
| robust_train | depth_fit | projected_rope | +0.012025 | [0.008953242134540858, 0.01552536094164067] |
| robust_train | depth_fit | expected_fixed | +0.008490 | [-0.005185482253099186, 0.02476629207917623] |
| robust_train | scale_noise | normalized_fixed | -0.035958 | [-0.04094256783570945, -0.03116787199656105] |
| robust_train | scale_noise | expected_shape | +0.145104 | [0.12809905756394255, 0.16169180677465972] |
| robust_train | scale_noise | projected_rope | -0.030096 | [-0.03615361730587109, -0.023720277087436577] |
| robust_train | scale_noise | expected_fixed | -0.036011 | [-0.04906116245167003, -0.02168086212601351] |
| robust_train | rgb_pred | normalized_fixed | +0.001164 | [-0.0012969425882934356, 0.003762940978925227] |
| robust_train | rgb_pred | expected_shape | -0.000777 | [-0.007147530719551557, 0.00590729958371827] |
| robust_train | rgb_pred | projected_rope | +0.009677 | [0.004212001082982562, 0.016103366883678385] |
| robust_train | rgb_pred | expected_fixed | +0.004289 | [-0.00809640368722365, 0.018521944234612297] |

All ten conditions, every seed, proper losses and calibration metrics remain in all_condition_means.json. The five principal conditions are independently replayed from archived model code, checked against saved test metrics and dataset digests; probability arrays and paired scene-bootstrap results are in point_summary.json. AP/ECE are summarized by seed rather than treating individual token rows as independent. This does not establish rejection calibration on real images, for which absent-track labels are not valid occlusion ground truth.
