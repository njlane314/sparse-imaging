import os


def _first(*names):
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _env(default, cast, *names):
    value = _first(*names)
    return default if value is None else cast(value)


ROOT_FILE = _first("ROOT_FILE") or "events.root"
TREE = _first("TREE") or "events"
BR_U = _first("BR_U") or "detector_image_u"
BR_V = _first("BR_V") or "detector_image_v"
BR_W = _first("BR_W") or "detector_image_w"
BR_Y = _first("BR_Y") or "is_signal"
BR_WGT = _first("BR_WGT") or "w_nominal"
STRICT_SHAPES = bool(_env(0, int, "STRICT_SHAPES"))
H = _env(512, int, "H")
W = _env(512, int, "W")
THRESH = _env(0.0, float, "THRESH")
BACKBONE = _first("BACKBONE") or "small"
EMBED_DIM = _env(256, int, "EMBED_DIM")
SHARD_EVENTS = _env(2048, int, "SHARD_EVENTS")
CHUNK_EVENTS = _env(256, int, "CHUNK_EVENTS")
MAX_BAD_EVENT_LOG = _env(25, int, "MAX_BAD_EVENT_LOG")
UPROOT_DECOMP_WORKERS = _env(2, int, "UPROOT_DECOMP_WORKERS")
FAULTHANDLER_TIMEOUT = _env(120, int, "FAULTHANDLER_TIMEOUT")
SHARDS_DIR = _first("SHARDS_DIR", "PROCESS_OUT_DIR", "SHARDS_OUT") or "shards"
PROCESS_OUT_DIR = SHARDS_DIR
SEED = _env(123, int, "SEED")
BATCH_SIZE = _env(32, int, "BATCH_SIZE", "BATCH")
NUM_WORKERS = _env(4, int, "NUM_WORKERS")
MAX_STEPS = _env(10000, int, "MAX_STEPS")
LR0 = _env(0.01, float, "LR0", "LR")
WEIGHT_DECAY = _env(1e-4, float, "WEIGHT_DECAY")
MOMENTUM = _env(0.9, float, "MOMENTUM")
POLY_POWER = _env(0.9, float, "POLY_POWER")
VAL_FRACTION = _env(0.1, float, "VAL_FRACTION", "VAL_FRAC")
VAL_EVERY = _env(2000, int, "VAL_EVERY")
VAL_BATCHES = _env(50, int, "VAL_BATCHES")
VAL_CACHE_BATCHES = _env(0, int, "VAL_CACHE_BATCHES")
CHECKPOINT_EVERY = _env(1000, int, "CHECKPOINT_EVERY")
CHECKPOINT_PATH = _first("CHECKPOINT_PATH", "OUT") or "checkpoints/checkpoint.pt"
LOSS_LOG_PATH = _first("LOSS_LOG_PATH") or "loss.tsv"
LOG_FLUSH_EVERY = _env(50, int, "LOG_FLUSH_EVERY")
TRAIN_DIAGNOSTICS_EVERY = _env(0, int, "TRAIN_DIAGNOSTICS_EVERY")
