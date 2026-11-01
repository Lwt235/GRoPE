# Fixed-query relative support conversion

10 checkpoints, all original-partition figurines/temple pairs. Max logit error 4.77e-06; max probability error 1.07e-06. Minimum encoder norm 0.184, above normalization epsilon.

CPU batch16, 48x48 pairs, seven repeats of100 forwards: {'fixed_query_reference': 5.411760751157999, 'relative_support': 3.753729797899723}; speedup 1.442x. The model retains15171 learned parameters and48-dimensional Q/K. This is an optional simplification of gaussian_konly; removing query covariance from gaussian_full is a separate empirical change, not proved lossless. Fitted query means remain necessary. Training with Adam after parameter conversion need not follow the original optimization trajectory.
