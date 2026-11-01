"""Run a python script with a hard per-process CUDA memory cap: capgpu.py GB script.py [args...]"""
import runpy, sys
import torch
gb = float(sys.argv[1]); sys.argv = sys.argv[2:]
if torch.cuda.is_available():
    tot = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / tot), 0)
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
