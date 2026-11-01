"""Contracts for frozen feature preparation, split isolation and array copying."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from experiments.real_frozen_features import (
    SCENES, array_sha256, export_scene, extract_pooled, fixed_projection,
    preprocess_image, project_features, sha256,
)


class FrozenVisualFeaturesTests(unittest.TestCase):
    def test_fixed_projection_orthogonality_normalization_and_no_rng_drift(self):
        state = torch.get_rng_state().clone()
        for width in [64, 128]:
            projection = fixed_projection(width)
            torch.testing.assert_close(projection, fixed_projection(width), atol=0, rtol=0)
            identity = projection @ projection.T if width == 64 else projection.T @ projection
            torch.testing.assert_close(identity, torch.eye(identity.shape[0]), atol=2e-6, rtol=2e-6)
            data = np.arange(48*width, dtype=np.float32).reshape(48, width)/width
            features = project_features(data, projection)
            self.assertEqual(features.shape, (48, 108))
            np.testing.assert_allclose(np.linalg.norm(features, axis=-1), 1, atol=2e-6)
        torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)

    def test_dense_forward_matches_official_torchvision_graph(self):
        from torchvision.models import resnet18
        from torchvision.models.feature_extraction import create_feature_extractor
        torch.set_num_threads(1)
        model = resnet18(weights=None).eval().requires_grad_(False)
        extractor = create_feature_extractor(model, return_nodes={'layer1':'layer1', 'layer2':'layer2'}).eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'image.png'
            image = np.random.default_rng(4).integers(0, 256, (81, 113, 3), dtype=np.uint8)
            Image.fromarray(image).save(path)
            prepared, _ = preprocess_image(path)
            actual, metadata = extract_pooled(path, model)
            with torch.inference_mode():
                dense = extractor(prepared[None])
            for layer, channels in [('layer1', 64), ('layer2', 128)]:
                expected = F.adaptive_avg_pool2d(dense[layer], (6, 8))[0]
                expected = expected.permute(1, 2, 0).reshape(48, channels).numpy()
                np.testing.assert_array_equal(actual[layer+'_mean'], expected)
            self.assertEqual(metadata['original_size'], [113, 81])
            self.assertEqual(metadata['normalized_input_sha256'], array_sha256(prepared.numpy()))

    def test_export_never_opens_test_files_and_preserves_nonfeature_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination, cache = root/'source', root/'output', root/'cache'
            source.mkdir()
            photo_paths = []
            for image_id in [1, 2]:
                path = root/f'{image_id}.png'
                Image.new('RGB', (32, 24), (image_id*30, 50, 70)).save(path)
                photo_paths.append(path)
            dataset = dict(images=[dict(id=1, path='1.png'), dict(id=2, path='2.png'),
                                   dict(id=3, path='UNREAD_TEST_IMAGE.png')])
            (root/'dataset.json').write_text(json.dumps(dataset))
            generator = np.random.default_rng(7)
            arrays = dict(q_features=np.zeros((48, 108), np.float32), k_features=np.ones((48,108),np.float32),
                          geometry=generator.normal(size=(48,3,3)), soft_labels=np.eye(48,dtype=np.float32),
                          label_valid=np.ones(48,bool), track_counts=np.eye(48,dtype=np.int32)*4)
            pairs = []
            for index, split in enumerate(['train', 'validation']):
                filename = f'pair_{index}_{split}.npz'
                np.savez_compressed(source/filename, **arrays)
                pairs.append(dict(split=split, q_id=1, k_id=2, file=filename, npz_sha256=sha256(source/filename)))
            pairs.append(dict(split='test', q_id=3, k_id=3, file='UNREAD_TEST_PAIR.npz', npz_sha256='never read'))
            manifest = dict(scene=SCENES[0], pairs=pairs, dataset=str(root/'dataset.json'),
                            available_frames=3, frames=[dict(image_id=i) for i in [1,2,3]],
                            source_images_sha256={str(path.resolve()):sha256(path) for path in photo_paths})
            (source/'manifest.json').write_text(json.dumps(manifest))
            pooled = dict(layer1_mean=generator.normal(size=(48,64)).astype(np.float32),
                          layer2_mean=generator.normal(size=(48,128)).astype(np.float32))
            projections = dict(layer1=fixed_projection(64), layer2=fixed_projection(128))
            with patch('experiments.real_frozen_features.extract_pooled', return_value=(pooled, {'fixture':True})) as extract:
                result = export_scene(source,destination,None,{'weights_sha256':'fixture'},projections,'layer2',cache)
            self.assertEqual(extract.call_count,2)
            self.assertEqual(result['pair_count'],2)
            self.assertEqual({record['split'] for record in result['pairs']},{'train','validation'})
            self.assertEqual([frame['image_id'] for frame in result['frames']],[1,2])
            self.assertFalse((destination/'UNREAD_TEST_PAIR.npz').exists())
            for pair in result['pairs']:
                with np.load(destination/pair['file'],allow_pickle=False) as saved:
                    for key,value in arrays.items():
                        if key not in ('q_features','k_features'):
                            np.testing.assert_array_equal(saved[key],value)
                    np.testing.assert_array_equal(saved['q_features'],project_features(pooled['layer2_mean'],projections['layer2']))
            # Exact settings reuse per-image cache without any feature forward.
            with patch('experiments.real_frozen_features.extract_pooled', side_effect=AssertionError('cache must be reused')):
                export_scene(source,destination,None,{'weights_sha256':'fixture'},projections,'layer2',cache)


if __name__ == '__main__':
    unittest.main()
