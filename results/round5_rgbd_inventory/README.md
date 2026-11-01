# Round5 RGB-D and frozen depth-model inventory

No qualifying real static RGB-D + camera-pose collection or monocular-depth checkpoint was found in the bounded local search. This does not claim absence outside the inspected roots/depth limits. No credential file contents were read; no model/data/environment was changed.

Local LERF/Middlebury collections contain RGB and SfM/reconstruction geometry, not new sensor-depth ground truth. The local NTU_RGB+D directory has RGB and skeletons but no observed depth stream or static-scene camera trajectory. SQA3D has videos without local ScanNet depth/pose assets. ResNet/SAM/OpenCLIP/VLM caches do not supply metric depth.

The preferred benchmark source is [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) with named scenes split between fitting, validation and testing. Its camera tracks are KinectFusion reference estimates. The [official 3DMatch prepared format](https://3dmatch.cs.princeton.edu/) explicitly uses aligned color/depth, millimeter depth and camera-to-world poses; it avoids silently assuming raw 7-Scenes RGB/depth alignment. Downloads may be gigabytes; network access belongs to the parent agent.

Smaller [UW RGB-D Scenes v2](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/) scene archives are available through 3DMatch, but scene-number versus physical-room separation must be verified. [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) has registered sensor depth and mocap trajectories; desk/desk2/xyz/rpy from the same office must not be treated as disjoint scenes. The five-frame [Open3D Redwood sample](https://www.open3d.org/docs/0.16.0/cpp_api/classopen3d_1_1data_1_1_sample_redwood_r_g_b_d_images.html) can validate projection/visibility code only.

For RGB-only geometry, the official [Depth Anything V2 metric indoor small model](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) is a compact candidate (24.8M parameters, Hypersim metric fine-tuning). It is not cached locally. Prediction must remain independent of sensor-depth buffers and cannot be scaled using held-out GT depth.

Inventory.json records exact local paths, limits, primary-source URLs, formats, sizes and caveats. No task scores or old GS data were substituted for a new-data experiment.
