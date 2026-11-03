"""Compare controllers on the representative sample, not the private full test set."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from simulate import ROOT
from v6.evaluate import run_clip

def main():
    rows=json.loads((ROOT/'release_artifacts/personal_hand_landmarks.json').read_text())['rows'];out=ROOT/'private/sample_evaluation';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'sample.npz',canonical=[r['canonical'] for r in rows],features=[r['features'] for r in rows],valid=[r['valid_cached'] for r in rows],timestamps=[r['source_timestamp_s'] for r in rows],frame_indices=[r['source_frame'] for r in rows])
    result,_,_=run_clip({'cache':'sample.npz','index':'representative_sample'},'sample',False,out)
    (out/'results.json').write_text(json.dumps({'scope':'Representative sample only; not reproduction of full held-out metrics; no training','results':result},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
