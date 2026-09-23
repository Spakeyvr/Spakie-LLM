"""Re-score saved BASE probe completions without regenerating or retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._base_probe_cases import SCORING_VERSION, score_completion


def rescore(records):
    groups, changes = {}, []
    for row in records:
        if row.get('kind') != 'generation' or 'expected_regex' not in row:
            continue
        passed = score_completion(row['output'], row)
        phase = row.get('phase', 'evaluation')
        metric = groups.setdefault(phase, {}).setdefault(row['category'],
                    {'passed':0, 'original_passed':0, 'total':0})
        metric['passed'] += int(passed)
        metric['original_passed'] += int(bool(row.get('passed')))
        metric['total'] += 1
        if passed != row.get('passed'):
            changes.append({'id':row['id'], 'phase':phase, 'category':row['category'],
                            'original_passed':row.get('passed'), 'passed':passed,
                            'prompt':row['prompt'], 'output':row['output']})
    return {'scoring_version':SCORING_VERSION, 'metrics':groups, 'changed_records':changes,
            'correction':'Numeric answers allow sentence-final periods and grouped thousands while rejecting different values and digit prefixes. Facts accept immediate capital-city and English playwright/poet descriptors. Original generation bytes remain unchanged.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserve earlier evidence and choose a new path')
    result = rescore(json.loads(line) for line in args.records.read_text().splitlines())
    result['input'] = str(args.records.resolve())
    result['input_sha256'] = hashlib.sha256(args.records.read_bytes()).hexdigest()
    result['scorer_sha256'] = hashlib.sha256((ROOT/'scripts/_base_probe_cases.py').read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['metrics']),flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
        raise SystemExit(130)
