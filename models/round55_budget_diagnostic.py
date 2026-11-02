"""Select frozen R52/R53/R54 model classes without changing their arithmetic."""
from models.round52_camera_frame import CameraFrameMatcher, INPUT_FIELDS
from models.round53_ray_frame import RayFrameMatcher
from models.round54_calibrated_plane import CalibratedPlaneMatcher

ROLES = ('camera_only','camera_gs','camera_offset','camera_plane',
         'ray_point','ray_iso','ray_gaussian','ray_affine')
GROUPS = (ROLES[:2], ROLES[2:4], ROLES[4:6], ROLES[6:])


def make_model(role):
    if role not in ROLES:raise ValueError(role)
    if role.startswith('ray_'):return RayFrameMatcher(role)
    if role in ('camera_offset','camera_plane'):return CalibratedPlaneMatcher(role)
    return CameraFrameMatcher(role)
