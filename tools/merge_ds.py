"""Merge LeRobot datasets (same features) into one: merge_ds.py OUT_ROOT SRC_ROOT..."""
import sys
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
out, srcs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
aggregate_datasets(repo_ids=[f"local/{p.name}" for p in srcs], aggr_repo_id=f"local/{out.name}", roots=srcs, aggr_root=out)
print("done", out)
