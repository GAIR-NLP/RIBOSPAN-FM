"""Filesystem roots used by the scripts in this directory.

REPO is the repository itself and is derived from this file's location, so it
needs no configuration. PROJ is the external storage that holds the pretraining
corpus, the pretrained checkpoints and the run outputs; it is machine-specific
and is read from the LIMBER_PROJ environment variable, falling back to a
``project`` directory beside the repository.

    export LIMBER_PROJ=/path/to/limber-storage
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROJ = Path(os.environ.get("LIMBER_PROJ", REPO.parent / "limber-project"))
