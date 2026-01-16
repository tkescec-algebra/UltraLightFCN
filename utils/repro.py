# utils/repro.py
import os
import random, numpy as np, torch

GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED, deterministic: bool = True, strict: bool = False) -> None:
    """
    deterministic=True  -> reproducibilno, ali ne nužno bit-exact.
    strict=True         -> pokušaj bit-determinističnog (sporije).
    """
    # 0) CUBLAS determinism mora biti postavljen prije prvog CUDA handle-a
    if deterministic:
        # :4096:8 je stabilniji ali troši više mem., :16:8 je štedljiviji
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8" if strict else ":16:8")
    else:
        # ako želiš eksplicitno maknuti restrikciju
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    # 1) RNG
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2) PyTorch/CuDNN/CUDA flags
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        try:
            torch.set_float32_matmul_precision("highest")
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass


def seed_worker(worker_id: int) -> None:
    """
    Ensure each DataLoader worker has a deterministic RNG state.
    This controls any numpy/random usage inside datasets/transforms.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
