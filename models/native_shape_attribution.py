"""The same radial control inside the unchanged native-GS classifier."""
from models.round73_native_relative5 import CompactNativeClassifier
from models.region_shape_attribution import fixed_hook

ROLES = ('fixed_isotropic5',)


class NativeFixedIsotropic(CompactNativeClassifier):
    def __init__(self, role='fixed_isotropic5'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position.mlp.register_forward_pre_hook(fixed_hook)
