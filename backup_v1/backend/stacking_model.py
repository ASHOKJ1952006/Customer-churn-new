import sys
from scripts.stacking_model import StackingEnsemble

# Register in sys.modules under both names for robust unpickling
sys.modules['stacking_model'] = sys.modules.get(__name__, sys.modules.get('scripts.stacking_model'))

__all__ = ["StackingEnsemble"]

