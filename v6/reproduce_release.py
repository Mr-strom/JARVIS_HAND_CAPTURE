"""Re-run the unchanged v6 evaluator on the portable held-out release pack."""
import json
from pathlib import Path
from v6.evaluate import run_clip
from simulate import ROOT

def main():
    dataset = ROOT / 'private/final/heldout_landmarks'
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    results = []
    for entry in manifest['split']:
        rows, _, _ = run_clip(entry, 'test', False, dataset)
        results.extend(rows)
    assert all(not sum(r['warnings'].values()) and not r['command_limit_violations'] and not r['joint_limit_violation_samples'] for r in results)
    expected = json.loads((ROOT / 'v6/test_results.json').read_text(encoding='utf-8'))['results']
    differences = []
    for row in results:
        old = next(r for r in expected if r['recording_id'] == row['recording_id'] and r['method'] == row['method'] and r['scenario'] == 'clean')
        differences.append({'recording': row['recording_id'], 'method': row['method'],
                            'tip_difference_mm': row['raw_fingertip_mm'] - old['raw_fingertip_mm'],
                            'physical_difference_mm': row['physical_fingertip_mm'] - old['physical_fingertip_mm'],
                            'solver_timeouts': row['solver_timeouts']})
    out = ROOT / 'private/v6'; out.mkdir(parents=True, exist_ok=True)
    report = {'results': results, 'differences_from_archived_clean_run': differences,
              'scope': 'Same five held-out recordings and evaluator. Wall-clock solver deadlines can cause machine-load-dependent fallbacks; timing is not deterministic.'}
    (out / 'release_reproduction.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(differences, indent=2))

if __name__ == '__main__':
    main()
