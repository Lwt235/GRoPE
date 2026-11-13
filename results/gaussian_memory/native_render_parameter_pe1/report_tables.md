# Complete fixed parameter-PE tables

All data are already used. P/S/G retain identical Render content/coverage; Previous_R/Previous_L are inherited matching32-place/2000step references.

[All metrics/regions/conditions and seed/place cells](scoring/all_summary_cells.csv) · [All query metrics](scoring/metrics.json) · [All paired relative intervals](scoring/summary.json)

## Absolute fixed endpoints

|Scope / views|Method|Missing ROI spatial|Missing ROI raw|Missing ROI bias²|Full whole AbsRel|
|---|---|---:|---:|---:|---:|
|development/original|Previous_L|0.022031336|0.075911967|0.053880631|0.188348491|
|development/original|Previous_R|0.020101690|0.071488941|0.051387251|0.195861751|
|development/original|P|0.020034626|0.076042979|0.056008352|0.206644480|
|development/original|S|0.020803280|0.070270871|0.049467592|0.194474495|
|development/original|G|0.020622693|0.068888101|0.048265409|0.192413201|
|development/heldout|Previous_L|0.024574790|0.078146876|0.053572086|0.204828616|
|development/heldout|Previous_R|0.023680216|0.075001790|0.051321574|0.207105518|
|development/heldout|P|0.023810852|0.084310780|0.060499929|0.229996356|
|development/heldout|S|0.024229225|0.077015910|0.052786685|0.215099146|
|development/heldout|G|0.024035146|0.074527571|0.050492425|0.209109606|
|development/pooled|Previous_L|0.023303063|0.077029421|0.053726358|0.196588553|
|development/pooled|Previous_R|0.021890953|0.073245365|0.051354412|0.201483635|
|development/pooled|P|0.021922739|0.080176880|0.058254141|0.218320418|
|development/pooled|S|0.022516252|0.073643391|0.051127139|0.204786820|
|development/pooled|G|0.022328919|0.071707836|0.049378917|0.200761404|
|train_all/original|Previous_L|0.011116713|0.012357786|0.001241073|0.037450162|
|train_all/original|Previous_R|0.006783330|0.007964592|0.001181262|0.038441137|
|train_all/original|P|0.006609739|0.007501306|0.000891567|0.035755574|
|train_all/original|S|0.006529168|0.007466600|0.000937432|0.037313252|
|train_all/original|G|0.006506454|0.007386611|0.000880157|0.036295215|
|train_all/heldout|Previous_L|0.014845128|0.020584401|0.005739273|0.057143933|
|train_all/heldout|Previous_R|0.008618255|0.013451296|0.004833041|0.058790295|
|train_all/heldout|P|0.008458495|0.012963634|0.004505138|0.057219972|
|train_all/heldout|S|0.008406942|0.013440055|0.005033113|0.059399877|
|train_all/heldout|G|0.008393445|0.013469983|0.005076538|0.058654553|
|train_original/heldout|Previous_L|0.016148864|0.020546431|0.004397567|0.047189930|
|train_original/heldout|Previous_R|0.007804878|0.011161296|0.003356419|0.049976859|
|train_original/heldout|P|0.007698072|0.010621738|0.002923666|0.047131769|
|train_original/heldout|S|0.007585314|0.011301734|0.003716420|0.050590339|
|train_original/heldout|G|0.007577389|0.011331472|0.003754083|0.049686581|
|train_added/heldout|Previous_L|0.014062887|0.020607183|0.006544296|0.063116334|
|train_added/heldout|Previous_R|0.009106282|0.014825296|0.005719014|0.064078356|
|train_added/heldout|P|0.008914750|0.014368771|0.005454022|0.063272893|
|train_added/heldout|S|0.008899919|0.014723048|0.005823129|0.064685601|
|train_added/heldout|G|0.008883079|0.014753090|0.005870010|0.064035336|

## Fixed comparisons, missing ROI spatial error

|Scope / views|Comparison|Absolute candidate−control|Relative reduction %|95% paired interval %|Three seed reductions %|
|---|---|---:|---:|---|---|
|development/original|G-P|0.000588066|-2.9352|[-10.1582, 1.2842]|-0.2883, -8.5632, -0.1691|
|development/original|G-S|-0.000180587|0.8681|[-1.8612, 4.5893]|3.3064, -1.2934, 0.7258|
|development/original|S-P|0.000768653|-3.8366|[-9.6481, 0.3917]|-3.7175, -7.1770, -0.9015|
|development/original|P-Previous_R|-0.000067064|0.3336|[-3.3949, 3.3221]|0.5335, 1.6792, -1.1071|
|development/original|S-Previous_R|0.000701590|-3.4902|[-9.3052, 1.4548]|-3.1643, -5.3773, -2.0185|
|development/original|G-Previous_R|0.000521003|-2.5918|[-8.9069, 2.3224]|0.2467, -6.7403, -1.2781|
|development/original|P-Previous_L|-0.001996710|9.0630|[-10.9106, 23.6333]|12.6107, 9.7715, 4.9463|
|development/original|S-Previous_L|-0.001228056|5.5741|[-17.2461, 20.9279]|9.3620, 3.2958, 4.0894|
|development/original|G-Previous_L|-0.001408643|6.3938|[-14.8509, 20.8125]|12.3588, 2.0450, 4.7855|
|development/heldout|G-P|0.000224295|-0.9420|[-7.7351, 2.7127]|0.0263, -4.0222, 0.9204|
|development/heldout|G-S|-0.000194078|0.8010|[-1.6274, 4.4187]|0.7678, 0.8523, 0.7845|
|development/heldout|S-P|0.000418373|-1.7571|[-9.1535, 1.9137]|-0.7472, -4.9164, 0.1370|
|development/heldout|P-Previous_R|0.000130636|-0.5517|[-4.5118, 3.3153]|-1.3200, 2.2381, -2.4567|
|development/heldout|S-Previous_R|0.000549009|-2.3184|[-7.2834, 2.9389]|-2.0771, -2.5683, -2.3164|
|development/heldout|G-Previous_R|0.000354930|-1.4988|[-5.8686, 3.1582]|-1.2934, -1.6941, -1.5137|
|development/heldout|P-Previous_L|-0.000763938|3.1086|[-19.9963, 26.7660]|5.4921, 6.6493, -2.9474|
|development/heldout|S-Previous_L|-0.000345565|1.4062|[-22.0952, 25.2934]|4.7859, 2.0598, -2.8064|
|development/heldout|G-Previous_L|-0.000539644|2.1959|[-20.8216, 25.6141]|5.5170, 2.8945, -1.9999|
|development/pooled|G-P|0.000406180|-1.8528|[-8.2767, 1.7748]|-0.1131, -6.1290, 0.4155|
|development/pooled|G-S|-0.000187333|0.8320|[-1.5622, 4.1486]|1.9111, -0.1546, 0.7572|
|development/pooled|S-P|0.000593513|-2.7073|[-8.7935, 1.0330]|-2.0636, -5.9652, -0.3442|
|development/pooled|P-Previous_R|0.000031786|-0.1452|[-3.9460, 3.0330]|-0.4901, 1.9796, -1.8269|
|development/pooled|S-Previous_R|0.000625299|-2.8564|[-7.6096, 2.1120]|-2.5639, -3.8675, -2.1774|
|development/pooled|G-Previous_R|0.000437966|-2.0007|[-6.6796, 2.6018]|-0.6038, -4.0281, -1.4037|
|development/pooled|P-Previous_L|-0.001380324|5.9234|[-15.3115, 24.5891]|8.7851, 8.1242, 0.8673|
|development/pooled|S-Previous_L|-0.000786811|3.3764|[-19.0969, 22.6128]|6.9027, 2.6437, 0.5261|
|development/pooled|G-Previous_L|-0.000974144|4.1803|[-17.3349, 22.7046]|8.6819, 2.4932, 1.2792|
|train_all/original|G-P|-0.000103286|1.5626|[0.0468, 3.1062]|2.1013, 1.4581, 1.1540|
|train_all/original|G-S|-0.000022714|0.3479|[-0.2222, 1.1360]|0.7018, -0.0481, 0.3892|
|train_all/original|S-P|-0.000080572|1.2190|[-0.2633, 2.7890]|1.4093, 1.5055, 0.7679|
|train_all/original|P-Previous_R|-0.000173591|2.5591|[1.3580, 4.2522]|1.8638, 1.8886, 3.8246|
|train_all/original|S-Previous_R|-0.000254162|3.7469|[2.0918, 5.2905]|3.2469, 3.3657, 4.5631|
|train_all/original|G-Previous_R|-0.000276876|4.0817|[2.4299, 5.6828]|3.9259, 3.3192, 4.9345|
|train_all/original|P-Previous_L|-0.004506973|40.5423|[33.4819, 47.4344]|43.6479, 39.9540, 37.8932|
|train_all/original|S-Previous_L|-0.004587545|41.2671|[34.2252, 47.9778]|44.4421, 40.8579, 38.3701|
|train_all/original|G-Previous_L|-0.004610259|41.4714|[34.4174, 48.3087]|44.8320, 40.8295, 38.6100|
|train_all/heldout|G-P|-0.000065050|0.7691|[-1.0338, 2.6094]|1.0804, 0.1729, 1.0507|
|train_all/heldout|G-S|-0.000013497|0.1605|[-0.6130, 1.0101]|0.7189, -0.4988, 0.2601|
|train_all/heldout|S-P|-0.000051553|0.6095|[-0.9639, 2.0987]|0.3642, 0.6684, 0.7927|
|train_all/heldout|P-Previous_R|-0.000159760|1.8537|[0.5490, 3.1490]|1.5445, 1.2429, 2.7477|
|train_all/heldout|S-Previous_R|-0.000211313|2.4519|[0.9831, 4.3921]|1.9030, 1.9030, 3.5185|
|train_all/heldout|G-Previous_R|-0.000224810|2.6085|[0.8116, 4.9015]|2.6082, 1.4137, 3.7695|
|train_all/heldout|P-Previous_L|-0.006386633|43.0217|[36.6858, 48.5118]|43.9489, 41.9904, 43.0937|
|train_all/heldout|S-Previous_L|-0.006438186|43.3690|[36.7064, 49.0045]|44.1530, 42.3781, 43.5448|
|train_all/heldout|G-Previous_L|-0.006451683|43.4599|[36.6756, 49.2341]|44.5545, 42.0907, 43.6916|
|train_original/heldout|G-P|-0.000120683|1.5677|[-1.9053, 3.4900]|2.0078, 2.1591, 0.5508|
|train_original/heldout|G-S|-0.000007925|0.1045|[-0.8432, 1.0467]|0.6622, -0.5542, 0.1813|
|train_original/heldout|S-P|-0.000112758|1.4648|[-1.7108, 3.2494]|1.3546, 2.6983, 0.3701|
|train_original/heldout|P-Previous_R|-0.000106806|1.3685|[-0.5956, 3.4847]|0.8409, -0.1409, 3.3053|
|train_original/heldout|S-Previous_R|-0.000219563|2.8132|[-0.3994, 5.3674]|2.1841, 2.5612, 3.6632|
|train_original/heldout|G-Previous_R|-0.000227488|2.9147|[-0.6824, 5.8744]|2.8318, 2.0212, 3.8379|
|train_original/heldout|P-Previous_L|-0.008450792|52.3306|[46.9194, 58.2368]|52.2787, 51.8397, 52.8510|
|train_original/heldout|S-Previous_L|-0.008563550|53.0288|[47.7264, 58.5018]|52.9251, 53.1392, 53.0255|
|train_original/heldout|G-Previous_L|-0.008571475|53.0779|[47.7804, 58.4683]|53.2369, 52.8795, 53.1107|
|train_added/heldout|G-P|-0.000031671|0.3553|[-1.9156, 2.9828]|0.5902, -0.8381, 1.3090|
|train_added/heldout|G-S|-0.000016840|0.1892|[-0.7180, 1.3392]|0.7484, -0.4715, 0.3011|
|train_added/heldout|S-P|-0.000014831|0.1664|[-1.9475, 2.3066]|-0.1594, -0.3649, 1.0110|
|train_added/heldout|P-Previous_R|-0.000191533|2.1033|[0.6414, 3.6322]|1.9124, 1.9327, 2.4569|
|train_added/heldout|S-Previous_R|-0.000206363|2.2662|[0.6328, 4.6178]|1.7560, 1.5748, 3.4431|
|train_added/heldout|G-Previous_R|-0.000223203|2.4511|[0.3948, 5.2867]|2.4913, 1.1108, 3.7338|
|train_added/heldout|P-Previous_L|-0.005148137|36.6080|[27.5795, 43.9273]|38.2509, 35.2495, 36.2784|
|train_added/heldout|S-Previous_L|-0.005162968|36.7134|[27.2098, 44.4971]|38.1525, 35.0132, 36.9227|
|train_added/heldout|G-Previous_L|-0.005179808|36.8332|[27.3786, 44.6420]|38.6153, 34.7068, 37.1126|

## Every development place, original queries / missing ROI spatial

|Place|Inherited Local|Inherited Render|P|S|G|G−P|G−S|S−P|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|brown_bm_1|0.033623607|0.038348918|0.040983187|0.042792935|0.041837625|0.000854438|-0.000955310|0.001809749|
|brown_bm_4|0.095073883|0.075636034|0.073820973|0.073326001|0.074651235|0.000830262|0.001325234|-0.000494972|
|brown_cogsci_1|0.033386314|0.041534292|0.041972411|0.043826683|0.041498980|-0.000473430|-0.002327702|0.001854272|
|brown_cs_3|0.003994172|0.003412603|0.003734478|0.004131848|0.004179345|0.000444867|0.000047497|0.000397370|
|harvard_robotics_lab|0.001808302|0.001525636|0.001415816|0.001784148|0.001994357|0.000578541|0.000210209|0.000368332|
|home_at|0.015397089|0.016133207|0.015260048|0.017342739|0.017198769|0.001938721|-0.000143970|0.002082691|
|home_bksh|0.026438173|0.026442012|0.025740798|0.027342812|0.027178593|0.001437795|-0.000164220|0.001602014|
|home_md|0.012957659|0.010705274|0.010829585|0.011031817|0.010854608|0.000025023|-0.000177208|0.000202231|
|mit_32_d507|0.003865214|0.002137895|0.002120678|0.002066442|0.002104933|-0.000015745|0.000038491|-0.000054236|
|mit_32_g725|0.002318094|0.002995571|0.002782712|0.002951275|0.003009599|0.000226887|0.000058324|0.000168563|
|mit_46_6conf|0.022203020|0.014484718|0.014264962|0.015245154|0.015184592|0.000919630|-0.000060562|0.000980192|
|mit_w59_conference|0.013310506|0.007864118|0.007489867|0.007797500|0.007779674|0.000289806|-0.000017826|0.000307633|

## Complete warm-query cost

|Method|Median ms|Peak GiB|
|---|---:|---:|
|Render|110.298|5.210|
|P|140.661|5.210|
|S|139.443|5.210|
|G|139.496|5.210|

Two fixed query conditions,5warmup+20measured each, seed20261110; exactly200 native calls total. Includes encoder, actual original content render, required PE fields/projection and native output. Excludes disk/JPEG/source construction/camera estimation; shared GPU timings are not exclusive deployment performance.

Bootstrap5000 paired seed/place resamples, seed20261113; equal queries/places; finite3seed/12development-place scope. Auxiliary comparisons remain correlated descriptive evidence. Material-harm rule is >=1% and supported CI; means/clip-generalization tradeoffs are reported, not zero-tolerance gates.
