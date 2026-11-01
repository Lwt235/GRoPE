# Frozen ResNet18 visual features: development only

The official torchvision ResNet18 architecture strictly loaded local ImageNet-1K V1 weights.
Weights SHA256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. No download, optimization or target-fitted projection was used.
Official checkpoint: https://download.pytorch.org/models/resnet18-f37072fd.pth

Only teatime / waldo_kitchen train and validation pairs and their photos were opened. No test pair/photo or external scene was opened.
Each image is converted to RGB, BOX-resized as a full image to 192x144, and ImageNet mean/std normalized. No crop is used.
The current pair features are frozen **layer2** dense average features over the existing row-major 8x6 region grid,
followed by a fixed seeded orthogonal projection to 108 dimensions and L2 normalization.
Layer2 uses 128->108 projection; layer1's optional 64->108 map is isometric before normalization.
Both layers' unprojected pooled vectors and projected alternatives are cached per image. The active layer is identical for every PE method.
Nonfeature pair arrays (geometry, cameras, labels, masks and track counts) are copied exactly and hashed individually.

This is supervised ImageNet classification pretraining, not a matching-trained visual backbone or a dense correspondence benchmark.
The official classifier's 256-resize / 224-center-crop evaluation recipe is deliberately not used: it would change the token/geometry grid.
Dense CNN receptive fields cross region boundaries; stride/padding make region pooling approximate image-region localization.
No claim about official ImageNet accuracy, stronger matching quality, or a universal positional encoding follows from this export.
The inherited sparse SfM target and reconstructed-GS limitations remain unchanged.

The cache records exact source-photo, normalized-input, feature-array, projection and weight hashes and can be reused only with identical settings.
These are development artifacts; a later confirmation export must freeze layer/projection/preprocessing first and explicitly authorize the additional splits.

Reproduce: `python -m experiments.real_frozen_features --source results/round3_real_support_data_adjacent --out results/round3_frozen_visual_data --layer layer2`
