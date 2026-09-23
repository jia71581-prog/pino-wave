#!/usr/bin/env python3
"""Recompute counterfactuals from existing per-record scalars; no dataset reads."""
import json, math, statistics
from pathlib import Path
p = Path(__file__).resolve().parent
energy = json.loads((p / 'ENERGY_PANEL.json').read_text())
rows = [json.loads(line) for line in (p / 'panel_29359_ROWS.jsonl').read_text().splitlines()]
result = {}
for family in ('uniform', 'layered', 'marmousi'):
    selected = [r for r in rows if r['medium_type'] == family]
    late, numer, denom, errors = [], 0.0, 0.0, []
    for row in selected:
        e = energy[row['sample_id']]
        assert e['medium_type'] == family and e['onset'] == row['onset']
        shares = [v / e['E_future'] for v in e['E_bands']]
        bands = row['time_bands']
        late.append(math.sqrt(shares[2]) * bands[2])
        numer += e['E_bands'][2] * bands[2] ** 2
        denom += e['E_future']
        errors.append(abs(sum(a * b * b for a, b in zip(shares, bands)) - row['future_relative_l2'] ** 2))
    result[family] = {'n': len(selected), 'record_equal_late_only': statistics.mean(late), 'energy_pooled_late_only': math.sqrt(numer / denom), 'identity_max_abs': max(errors)}
assert max(v['identity_max_abs'] for v in result.values()) < 2e-15
print(json.dumps(result, indent=2))
