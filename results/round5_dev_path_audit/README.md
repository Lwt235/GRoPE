# Development-only V-path audit

Uses the frozen normalized development checkpoint (seed101), first10 validation pairs and first8 train pairs only. No formal test data, optimizer update, architecture selection or confirmation model is used.

Maximum validation logit/probability change when zeroing V: 17.2209 / 0.862577. Full V-gradient norms by layer: [1.6536288261413574, 0.3763786554336548]; disabled-path V/output gradients are exactly zero. All model parameters/buffers and checkpoint bytes remain unchanged.

Pooled10-pair descriptive metrics: {'ordinary': {'n_valid': 2249, 'ce_sum': 1.6932016261955802, 'pck8_count': 0.8963983992885727, 'pck4_count': 0.5473543797243219, 'epe_argmax_sum': 4.708440554838514, 'epe_soft_sum': 5.772912103346693}, 'zero_values': {'n_valid': 2249, 'ce_sum': 4.259129064621723, 'pck8_count': 0.4522009782125389, 'pck4_count': 0.20809248554913296, 'epe_argmax_sum': 25.157595606956047, 'epe_soft_sum': 45.23320593149801}}. These quantify path activity, not generalization or PE-specific attribution.
