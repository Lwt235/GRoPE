# R66 classification diagnostic — 2026-11-04

User resumed the research objective. This is a prospective diagnostic of used
development data, not a replacement for R66 or fresh confirmation.

Question: why does CE-selected log_shape_distance improve object-average CE but
reduce mean-class accuracy against distance_mlp on all three dev_eval seeds?
Competing explanations are probability-vs-decision changes, class weighting,
training-path differences, and failure to generalize fitted shape relations.
The representational inclusion of distance does not guarantee the trained
extension's accuracy. The original zero-input-column initialization already
matched the distance model's initial function and training streams.

Fixed comparison: original R66 log_shape_distance versus R65 distance_mlp,
seeds 11/22/33, all 60 classes and both sources. Read saved selected validation
predictions and all six existing dev_eval conditions. Report object-average
and class-average CE, OA, mCA, true-label probability, true-vs-other logit margin,
and four mutually exclusive correct/correct, wrong/wrong, corrected, harmed
groups. Group contributions must add to the paired aggregate CE/accuracy gaps.
Report every class/source, without favorable-subgroup selection. Sources have
disjoint categories; source differences are descriptive, not causal.

Compare all 120 paired training losses and all 25 original validation points.
Training loss includes augmentation and 0.1 label smoothing; validation CE
does not. Do not subtract these as a generalization gap. For an aligned check,
evaluate all 6227 fit and 765 val objects, first 512 canonical cached tokens,
eval mode, no augmentation or label smoothing. Use both original CE-selected
weights and fixed terminal epoch120 weights; do not select among these policies.
That is 12 fixed models and 24 forward evaluations. Only the original selected
models' previously saved dev_eval predictions are analyzed. No new terminal
dev_eval scoring, training, optimization, temperature calibration or selection.

Validate original source/data/checkpoint/prediction identities and paired
memberships. Check independent metric agreement and additive decompositions.
Use idle GPUs for forward evaluation and preserve other users' work. Keep raw
results and a concise report; numerical checks should be proportional to effects.

Decisions: if changes in correct-label probabilities account for lower CE while
net classification flips lower mCA, report that decomposition without calling
it calibration improvement or a classification success. If lower aligned fit
loss is paired with worse validation, record generalization evidence, not proof
of a unique cause. If training path remains an actionable uncertainty, the
previously discussed single matched warm-start comparison may follow under its
own fixed short budget. Otherwise proceed to the cross-view mechanism study.
Explaining every past failure is not a prerequisite for that study.
