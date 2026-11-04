# R67: one matched late-shape training diagnostic

2026-11-04. The user resumed autonomous validation. R66 remains a negative
development result; its sources, predictions and selection are unchanged.

The preceding fixed diagnostic found lower object- and class-average CE but
lower OA and mCA on all three dev_eval seeds. In each seed, CE improvement on
objects whose correctness did not change offsets a positive CE contribution
from the corrected/harmed group. Net harmed objects are 4/6/7. Thus class
weighting alone does not explain the discrepancy. Aligned fit/val checks and
paired training curves have mixed signs across seeds; a single uniform claim
of optimization failure or overfitting is unsupported.

Question: after learning center relations, does adding the R66 shape relation
provide useful extra classification capacity under a matched short budget?
Competing explanations: late introduction helps the training path; shape again
improves probability scores without improving decisions; extra relations have
no useful marginal contribution for this task or do not generalize.

Exactly two roles, seeds11/22/33: distance_continue and shape_after_distance.
Each starts from the same seed's R65 distance_mlp terminal120 resume.pt,
including original common optimizer states and sampler state. The shape role
adds only the R66 log-shape input, initially zero; content13, backbone, head and
all previous distance parameters are identical. The first layer is written
W_r*r + W_s*s + b, algebraically the same two-input MLP, to give W_s its own
Adam state initialized at zero steps. All old Parameter objects and optimizer
entries remain in place. This adds96 parameters, not another network module
or learned gate. No spectrum role or new shape formula is introduced.

Both roles receive20 extra epochs,7800 updates, batch16,512tokens, the original
SO(3)/jitter augmentation, weight decay.01, label smoothing.1 and clip1.
Use one fixed cosine learning rate from1e-4 to1e-5 over the extra7800 updates;
this is a declared short fine-tuning schedule, not continuation of the old
120-epoch cosine beyond its endpoint. Both share it, with no LR search.
Resume the original sampler, use identical per-step dropout seeds, and record
all20 actual input-stream hashes. Validate at added epochs0/5/10/15/20 and
select earliest strict minimum validation CE, including the shared epoch0.
Select all six models before new dev_eval predictions. A selected epoch0 means
no accepted adaptation under this rule; report that explicitly.

Reuse6227fit/765val/765dev_eval, all60 classes and two sources. Export the
original six conditions for selected models only. The primary comparison is
shape_after_distance minus distance_continue base mCA, not a140-epoch model
versus an unextended120-epoch baseline. Report all seeds, class/source metrics,
OA, CE and class-average CE, selected epochs and cost. Keep original selected
R66 and R65 results as context, not alternative controls or selection choices.

Use the previous descriptive development lead rule (mean>=.5pp, all three
paired seeds positive, paired-class95% lower endpoint>0); also show effects
and uncertainty continuously. This is not a universal test gate and small
secondary regressions do not automatically veto a useful scoped result.
Class bootstrap averages the paired seeds before resampling60 classes and
does not estimate a seed population. No fixed-final dev_eval rescue scoring.

If warm shape helps, it supports this specified late-introduction pipeline
under equal extra budget; it does not prove an intrinsic optimization cause,
or superiority to a separately tuned140-epoch from-scratch shape model.
If only CE improves, classification utility remains unestablished. If no
stable mCA gain appears, end this one diagnostic and proceed to the planned
cross-view mechanism study. Do not search training rates, budgets or gates.
No officialTEST/reserved confirmation access; no general-effectiveness claim.

Proportional checks: original checkpoint/source/data identity, initial function
and optimizer-state equality, an inactive-shape paired update, equivalence to
the existing R66 formula with nonzero shape weights, active finite gradients,
paired actual input streams, independent classification metrics and plot
inspection. Use only GPUs observed idle and preserve other users' processes.
