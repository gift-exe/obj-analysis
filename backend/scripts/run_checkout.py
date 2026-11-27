"""Run the checkout counter once so its per-frame labels are cached for the app."""
import sys, time
from app.tools.registry import execute_gpu_tool, load_all
from app.core.config import get_settings
from app.tools.plugins.checkout import VERSION
load_all()
vid = sys.argv[1]
t = time.time()
r = execute_gpu_tool("checkout_scan_count", vid, {"version": VERSION, "fps": get_settings().checkout_fps})
print("done in", round(time.time() - t), "s", r.get("status"), r.get("tool_run_id"))
