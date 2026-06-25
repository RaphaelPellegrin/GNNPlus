from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

# Register custom train modes with GraphGym.
from . import custom_train  # noqa: F401
from . import heterogeneity_train  # noqa: F401
