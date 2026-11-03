"""Reanalyze saved real-camera telemetry without opening a camera or altering raw results."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from live_mirror import response_lag

source=ROOT/'reports/phase2_live1_metrics.json'
raw=json.loads(source.read_text())
with np.load(ROOT/'private/phase2_live1_telemetry.npz') as archive:
    samples=archive['samples']
analysis={'source_run':str(source.relative_to(ROOT)),
          'motion_response_lag_corrected':response_lag(samples),
          'approximate_frame_to_render_latency_ms':raw['capture_read_complete_to_render_ms'],
          'camera_frame_period_ms':1000/raw['webcam_fps'],
          'measurement_limits':'Read-completion to render is measured. Actual exposure-to-display motion latency is not measured; filter, slew limiting, camera buffering and screen scanout add variable delay.',
          'tracked_telemetry_rows':len(samples),
          'target_motion_is_live':raw['source'],
          'notes':['The raw live1 JSON is retained unchanged. Its old zero-lag bin must not be reported as zero latency.',
                   'Only tracked frames were sampled. Interpolation across occlusions makes motion correlation exploratory, not a latency benchmark.']}
(ROOT/'reports/phase2_live1_analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
print(json.dumps(analysis,indent=2))
