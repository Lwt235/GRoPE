"""Data-free algebra witness; never changes the frozen R52 experiment."""
import numpy as np
from scipy.linalg import expm
from experiments.round52_data import OUT, write, sha


def frame(linear, center):
    g = np.eye(4); g[:3, :3] = linear; g[:3, 3] = center
    return g


def main():
    destination = OUT/'operator_scope/result.json'
    if destination.exists():
        raise FileExistsError('retain algebra witness')
    rot = expm(np.array([[0., -.3, .2], [.3, 0, -.1], [-.2, .1, 0]]))
    K = np.diag([.9, 1.2, 1., 1.])
    p = K.copy(); p2 = K @ np.linalg.inv(frame(rot, [.1, -.2, .3]))
    centers = ([.1, .3, .8], [-.4, .2, 1.])
    shapes = (np.eye(3), np.diag([.2, .7, 1.4]), rot @ np.diag([.2, .7, 1.4]) @ rot.T)
    forward = []; inverse = []; shared_frame_errors = []
    expected = frame(np.eye(3), np.array(centers[0])-centers[1])
    for shape in shapes:
        g1, g2 = [frame(shape, mu) for mu in centers]
        forward.append((p @ g1) @ np.linalg.inv(p2 @ g2))
        inverse.append(np.linalg.inv(g1) @ g2)
        shared_frame_errors.append(float(np.max(np.abs((p @ g1) @ np.linalg.inv(p2 @ g1)-p @ np.linalg.inv(p2)))))
        np.testing.assert_allclose(g1 @ np.linalg.inv(g2), expected, atol=2e-14)
    forward_difference = float(np.max(np.abs(np.array(forward)-forward[0])))
    inverse_difference = float(np.max(np.abs(np.array(inverse)-inverse[0])))
    assert forward_difference < 2e-14 and inverse_difference > .1
    write(destination, dict(source_sha256=sha(__file__),data_access=False,
        shared_shape_forward_difference=forward_difference,
        shared_shape_inverse_difference=inverse_difference,
        common_G_relative_camera_errors=shared_frame_errors,
        statement='Shared L cancels from G_i G_j^-1 even for different centers; it does not cancel from G_i^-1 G_j translation.',
        scope='Algebraic scope only, not a claim that this causes learned-task errors; no trial or criterion changed.'))
    (destination.parent/'source.py').write_bytes(open(__file__,'rb').read())
    print('SCOPE', forward_difference, inverse_difference, shared_frame_errors, flush=True)


if __name__ == '__main__':
    main()
