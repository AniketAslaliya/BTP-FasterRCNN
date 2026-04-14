# model/faster_rcnn.py
# ---------------------------------------------------------------------------
# Backward-compatibility router.
# Legacy scripts (tools/train_with_eval.py, tools/full_evaluation.py, etc.)
# that do `from model.faster_rcnn import FasterRCNN` will continue to import
# the V1 (baseline) model unchanged.
#
# The new unified trainer (tools/train.py) uses its own dynamic factory
# `build_model(version)` and does NOT use this file.
# ---------------------------------------------------------------------------
from model.faster_rcnn_v1 import FasterRCNN  # noqa  (legacy default = V1)
