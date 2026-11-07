#!/usr/bin/env python3
"""Derive report-only inputs from completed, independently reviewed CSVs.

Run from any directory with Python 3.11+. Does not import experiment code,
train, evaluate, or modify frozen experiment files. Raw inputs stay outside Git.
Outputs are confined to report/generated; the manifest binds source and outputs.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

RESEARCH = Path(__file__).resolve().parent.parent
PROJECT = RESEARCH.parent
COPIED = PROJECT / 'local-materials/report-inputs/final-review'
INPUT = COPIED if COPIED.is_dir() else PROJECT / 'experiment-results/final'
AUDITS = PROJECT / 'local-materials/audits'
OUTPUT = RESEARCH / 'report/generated'
REMOTE = '/home/ssdd/academics/comp597/experiment-results/final/'
REGIMES = ('static_clean', 'changing_clean', 'changing_missed')
POLICIES = ('reactive', 'ttl', 'reset', 'voi', 'ppo_full', 'ppo_masked')
LABEL = {'static_clean': 'Static clean', 'changing_clean': 'Changing clean',
         'changing_missed': 'Changing with misses', 'reactive': 'Reactive',
         'ttl': 'Route-conditioned TTL', 'reset': 'Memory reset', 'voi': 'VOI',
         'ppo_full': 'Full PPO', 'ppo_masked': 'Masked PPO'}
METRICS = ('success_fraction', 'time_per_task', 'inspections_per_task',
           'blocked_attempts', 'reward', 'penalized_cost')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8), (a, b)


def main():
    audit_path = AUDITS / 'final-results-review.json'
    train_path = AUDITS / 'final-training-review.json'
    audit = json.loads(audit_path.read_text())
    train = json.loads(train_path.read_text())
    assert audit['verdict'] == train['verdict'] == 'PASS'
    assert not audit['remaining_blockers'] and not train['blockers']
    assert audit['source_sha256'] == train['expected_source_sha256']
    assert audit['config_sha256'] == train['expected_config_sha256']
    names = ['evaluation/sequences.csv', 'evaluation/episodes.csv',
             'evaluation/summary.json', 'evaluation/provenance.json',
             'aggregate/per_seed_summary.csv', 'aggregate/training_seed_variation.csv',
             'aggregate/paired_differences.csv', 'aggregate/learning_curve.csv',
             'aggregate/learning_per_seed.csv', 'aggregate/learning_curve.png']
    inputs = {}
    for name in names:
        digest = sha(INPUT / name)
        assert digest == audit['output_sha256'][REMOTE + name], name
        inputs[REMOTE + name] = digest
    for path in (audit_path, train_path):
        inputs['../local-materials/audits/' + path.name] = sha(path)

    sequences = read_csv(INPUT / 'evaluation/sequences.csv')
    tasks = read_csv(INPUT / 'evaluation/episodes.csv')
    per_seed = read_csv(INPUT / 'aggregate/per_seed_summary.csv')
    paired = read_csv(INPUT / 'aggregate/paired_differences.csv')
    seed_sd = read_csv(INPUT / 'aggregate/training_seed_variation.csv')
    curves = read_csv(INPUT / 'aggregate/learning_curve.csv')
    windows = read_csv(INPUT / 'aggregate/learning_per_seed.csv')
    assert (len(sequences), len(tasks), len(per_seed), len(paired), len(curves), len(windows)) == (5376, 32256, 252, 108, 96, 480)
    by_task = defaultdict(list)
    by_seed = defaultdict(list)
    by_scenario = defaultdict(dict)
    for row in tasks:
        by_task[(row['policy'], row['training_seed'], row['scenario_id'])].append(row)
    for row in sequences:
        key = (row['policy'], row['training_seed'], row['scenario_id'])
        members = by_task.pop(key)
        assert sorted(int(x['task']) for x in members) == list(range(6))
        for metric in ('success', 'failure', 'time', 'inspections', 'blocked_attempts', 'reward', 'penalized_cost'):
            close(row[metric], math.fsum(float(x[metric]) for x in members))
        close(row['reward'], 600 - float(row['time']) - 200 * float(row['failure']))
        close(row['penalized_cost'], float(row['time']) + 200 * float(row['failure']))
        by_seed[(row['policy'], row['training_seed'], row['regime'])].append(row)
        scenario = by_scenario[row['scenario_id']]
        assert (row['policy'], row['training_seed']) not in scenario
        scenario[(row['policy'], row['training_seed'])] = row
    assert not by_task and len(by_scenario) == 384
    expected = {(p, 'baseline') for p in POLICIES[:4]} | {(p, str(s)) for p in POLICIES[4:] for s in range(5)}
    for rows in by_scenario.values():
        assert set(rows) == expected
        ref = rows[('voi', 'baseline')]
        for policy in POLICIES[4:]:
            for seed in range(5):
                for metric in METRICS:
                    close(rows[(policy, str(seed))][metric], ref[metric])
    safe_records = [r for r in tasks if r['policy'] in ('voi', 'ppo_full', 'ppo_masked')]
    assert len(safe_records) == 25344
    assert all(int(r['success']) == 1 and int(r['inspections']) == int(r['blocked_attempts']) == 0 for r in safe_records)

    lookup = {}
    for row in per_seed:
        values = [float(x[row['metric']]) for x in by_seed[(row['policy'], row['training_seed'], row['regime'])]]
        assert len(values) == int(row['n']) == 128
        close(row['mean'], statistics.mean(values))
        close(row['sd'], statistics.stdev(values))
        lookup[(row['policy'], row['training_seed'], row['regime'], row['metric'])] = float(row['mean'])
    for row in seed_sd:
        means = [lookup[(row['policy'], str(s), row['regime'], row['metric'])] for s in range(5)]
        close(row['sd'], statistics.stdev(means))
    for row in paired:
        delta = []
        for sid, rows in by_scenario.items():
            if f"test/{row['regime']}/" not in sid:
                continue
            values = [float(v[row['metric']]) for (p, _), v in rows.items() if p == row['policy']]
            delta.append(statistics.mean(values) - float(rows[('reactive', 'baseline')][row['metric']]))
        assert len(delta) == int(row['n']) == 128
        mean, sd = statistics.mean(delta), statistics.stdev(delta)
        close(row['mean'], mean)
        close(row['normal_95_low'], mean - 1.96 * sd / math.sqrt(128))
        close(row['normal_95_high'], mean + 1.96 * sd / math.sqrt(128))
    for row in curves:
        values = [float(w['mean_raw_return']) for w in windows if w['policy'] == row['policy'] and w['window_end_env_steps'] == row['window_end_env_steps']]
        assert len(values) == int(row['n']) == 5
        close(row['mean'], statistics.mean(values)); close(row['sd'], statistics.stdev(values))
    assert all(max(int(r['window_end_env_steps']) for r in curves if r['policy'] == p) == 98304 for p in POLICIES[4:])

    def value(policy, regime, metric):
        seeds = range(5) if policy.startswith('ppo_') else ('baseline',)
        return statistics.mean(lookup[(policy, str(s), regime, metric)] for s in seeds)

    lines = [r'% Generated from audited CSVs by report/generate_results.py.',
             r'\begin{table*}[t]',
             r'\caption{Held-out graph results. Each regime uses 128 paired six-task sequences. PPO entries average five independently trained models; classical policies have one deterministic instance. $T$: failure-inclusive graph time/task; $I$: inspection attempts/task; $B$: blocked attempts/task; $C=T+200(1-S/100)$: penalized cost/task. Cost SD is across the five model means, not sequence uncertainty. All values are derived from saved CSVs.}',
             r'\label{tab:results}\centering\small',
             r'\begin{tabular}{llrrrrrr}\hline',
             r'Regime & Policy & $S$ (\%) & $T$ & $I$ & $B$ & $C$ & Cost seed SD \\ \hline']
    for regime in REGIMES:
        for index, policy in enumerate(POLICIES):
            vals = [100 * value(policy, regime, 'success_fraction'), value(policy, regime, 'time_per_task'),
                    value(policy, regime, 'inspections_per_task'), value(policy, regime, 'blocked_attempts') / 6,
                    value(policy, regime, 'penalized_cost') / 6]
            for field, actual in zip(('success_percent','all_task_time','inspections_per_task','blocked_attempts_per_task','penalized_cost_per_task'), vals):
                close(actual, audit['report_ready_metrics'][regime][policy][field])
            spread = 'n/a'
            if policy.startswith('ppo_'):
                spread = f"{statistics.stdev(lookup[(policy,str(s),regime,'penalized_cost')]/6 for s in range(5)):.2f}"
            lines.append(f"{LABEL[regime] if index == 0 else ''} & {LABEL[policy]} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.3f} & {vals[3]:.3f} & {vals[4]:.2f} & {spread} " + r'\\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}', r'\end{table*}']
    paired_lines = [r'% Generated from paired_differences.csv; sequence costs divided by six.',
                    r'\begin{table}[t]',
                    r'\caption{Full PPO minus reactive penalized cost per task. Negative differences favor PPO on this penalized metric, not raw time. Intervals use 128 paired sequences per regime, conditional on the five models, and are exploratory and unadjusted.}',
                    r'\label{tab:paired}\centering\small',
                    r'\begin{tabular}{lrr}\hline Regime & Mean & Normal 95\% CI \\ \hline']
    for regime in REGIMES:
        row = next(r for r in paired if r['policy'] == 'ppo_full' and r['regime'] == regime and r['metric'] == 'penalized_cost')
        mean, low, high = [float(row[k])/6 for k in ('mean','normal_95_low','normal_95_high')]
        paired_lines.append(f'{LABEL[regime]} & ${mean:.2f}$ & $[{low:.2f}, {high:.2f}]$ ' + r'\\')
    paired_lines += [r'\hline\end{tabular}', r'\end{table}']
    runtime = json.loads((INPUT/'evaluation/provenance.json').read_text())['runtime']
    timing = train['timing']
    assert train['summary']['verified_runs'] == 10 and train['summary']['steps_per_run'] == 98304
    close(sum(timing['per_run_seconds']), timing['sum_recorded_training_seconds'])
    compute = ('% Generated from the PASS training audit and evaluation runtime provenance.\n'
               f"Training used CUDA on an NVIDIA GeForce RTX 5090, with PyTorch {runtime['torch']}, Python {runtime['python'].split()[0]}, and NumPy {runtime['numpy']}. "
               f"The ten recorded training-loop times sum to {timing['sum_recorded_training_seconds']:.6f} seconds; individual loops range from {timing['minimum_seconds']:.3f} to {timing['maximum_seconds']:.3f} seconds. "
               'These timers include rollout collection, PPO updates, and in-loop logging. They exclude initial provenance creation, model/optimizer setup, and initial environment construction, as well as final checkpoint and artifact serialization. '
               'They are not end-to-end process timings or evidence of real-time navigation. No independently captured batch process timer or separate inference-latency measurement is claimed.\n')
    figure = r'''% Verified aggregate PNG and its CSV inputs are recorded in results-manifest.json.
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{generated/learning_curve.png}
\caption{Graph-only PPO training. Each point averages five independent seeds; shading is $\pm1$ sample standard deviation of seed means, not a confidence interval over held-out sequences. Fixed 2,048-step windows use completed raw six-task sequence returns, ending at 98,304 steps. There are 48 points per variant, 96 total. Partial trailing sequences are excluded. Improved return does not demonstrate learned inspection.}
\label{fig:learning}
\end{figure}
'''
    OUTPUT.mkdir(exist_ok=True)
    generated = {'metrics.tex': '\n'.join(lines)+'\n', 'paired.tex': '\n'.join(paired_lines)+'\n',
                 'compute.tex': compute, 'learning-curves.tex': figure}
    for name, content in generated.items():
        (OUTPUT/name).write_text(content)
    shutil.copyfile(INPUT/'aggregate/learning_curve.png', OUTPUT/'learning_curve.png')
    outputs = {name: sha(OUTPUT/name) for name in (*generated, 'learning_curve.png')}
    manifest = {'status': 'REVIEW-CANDIDATE; independent report audit pending',
                'scientific_results_review': 'PASS', 'training_review': 'PASS',
                'generator': 'report/generate_results.py', 'generator_sha256': sha(Path(__file__)),
                'experiment_source_sha256': audit['source_sha256'], 'experiment_config_sha256': audit['config_sha256'],
                'input_sha256': inputs, 'output_sha256': outputs,
                'checks': {'sequence_rows': len(sequences), 'task_rows': len(tasks), 'curve_points': len(curves),
                           'seed_windows': len(windows), 'voi_full_masked_tasks_without_inspection': len(safe_records),
                           'csv_statistics_recomputed': True, 'paired_intervals_recomputed': True,
                           'plot_png_matches_reviewed_aggregate': True}}
    (OUTPUT/'results-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(manifest['checks'], indent=2))
    print('Generated report-only inputs and results-manifest.json. No experiment files changed.')


if __name__ == '__main__':
    main()
