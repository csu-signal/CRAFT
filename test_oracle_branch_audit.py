"""Branch coverage for the CRAFT oracle enumerator.

Run from the repository root:
    python -m unittest -v test_oracle_branch_audit

test_oracle.py only reaches states produced by applying oracle moves to an empty
board, so it can never construct a cell whose lower layers disagree with the
target. These tests build those states directly and exercise all four branches
of enumerate_correct_actions, then measure what a prefix-guard fix would change.

Layout:
  CurrentBehavior  characterizes the enumerator as it stands today, defects included
  CandidateDelta   current function vs an isolated prefix-check variant
  KnownGaps        desired invariants the code does not yet satisfy (expectedFailure)
  DatasetChecks    whole-dataset sweeps over data/structures_dataset_20.json
  ProgressChecks   per-move and net change in the environment's overall_progress

Real project modules throughout — no mock simulator or replacement environment.
agents.builder_tools.simulate_move and agents.environment.EnhancedGameState are
the production objects. The only scaffolding is the openai import shim below,
which touches nothing in the board logic or the oracle.

The candidate variant is a local stack-repair fix, NOT a general large-block
planner; test_local_prefix_fix_does_not_resolve_cross_cell_blockers records
where it stops. Dataset path overridable via the CRAFT_DATASET env var.

Current expectations: 29 tests, 21 passing and 8 expected failures. An expected
failure is visible debt, not a pass — the suite going green does not mean those
gaps are closed. Figures: 6/20 structures build exactly under default rules,
14 targets carry forbidden spans, and the injection sweep repairs only
top-of-stack-at-full-depth. 141 injections cover all-small-block cells only,
with other cells already at target; candidate lists match on 103 clean states.

When the prefix guard lands in agents/oracle.py: candidate_function() raises
RuntimeError by design once that source changes, so delete it, repoint
CandidateDelta/DatasetChecks at the real enumerator, and drop the
expectedFailure from the two gaps the fix closes.
"""
import copy
import json
import os
import sys
import types
from pathlib import Path
from collections import defaultdict
import inspect
import random
import textwrap
import unittest

# ── Dependency shim ───────────────────────────────────────────────────────────
# agents/environment.py binds the OpenAI client at module load time, but nothing
# in this suite makes a network call — it only exercises the board logic and the
# oracle enumerator. If the SDK is absent (a laptop checkout rather than the run
# environment), register a stub so the import chain resolves. When the real SDK
# is installed this block does nothing.
try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("openai")

    class _OfflineClient:
        """Placeholder — constructing it is fine, calling the API is not."""
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            raise RuntimeError(
                "test_oracle_branch_audit.py runs offline; no OpenAI calls expected "
                f"(attempted to access client.{name})"
            )

    _stub.OpenAI = _OfflineClient
    _stub.AzureOpenAI = _OfflineClient
    sys.modules["openai"] = _stub

from agents.environment import EnhancedGameState, get_oracle_moves
from agents.oracle import enumerate_correct_actions
import agents.oracle as oracle_module

P, Q, R, S = '(0,0)', '(1,0)', '(0,1)', '(1,1)'
COORDS = [f'({i},{j})' for i in range(3) for j in range(3)]


def board(updates=None):
    result = {p: [] for p in COORDS}
    result.update(copy.deepcopy(updates or {}))
    return result


def game(target, current=None, target_spans=None, current_spans=None, **kwargs):
    kwargs.setdefault('invisible_cells', [])
    kwargs.setdefault('partType', 'empty')
    g = EnhancedGameState(copy.deepcopy(target),
                          target_spans=copy.deepcopy(target_spans or {}), **kwargs)
    if current is not None:
        g.current_structure = copy.deepcopy(current)
        g.current_spans = copy.deepcopy(current_spans or {})
    return g


def ok_entries(g, fn=enumerate_correct_actions):
    return [e for e in fn(g) if e['flag'] == 'ok']


def execute(g, move):
    result = g.execute_move(copy.deepcopy(move))
    if not result[0]:
        raise AssertionError(f'Oracle move failed: {move}: {result[1]}')
    return result


def canonical_spans(spans):
    return {int(k): sorted(tuple(sorted(pair)) for pair in pairs)
            for k, pairs in spans.items() if pairs}


def exact(g):
    return (board(g.current_structure) == board(g.target_structure)
            and canonical_spans(g.current_spans) == canonical_spans(g.target_spans))


def signature(g):
    return repr((g.current_structure, canonical_spans(g.current_spans)))


def progress(g):
    """Environment's own composite score: (iou + completion + position_accuracy) / 3."""
    return g.progress_tracker.calculate_progress(g.current_structure)['overall_progress']


def blocks_correct(g):
    return g.progress_tracker.calculate_progress(g.current_structure)['blocks_placed_correctly']


def injection_cases(data):
    """Yield (structure_id, target, spans, pos, current_stack, label) over all-small-block cells.

    Mirrors DatasetChecks.test_corruption_delta_and_clean_prefix_equivalence so the
    progress report is measured on exactly the states the delta table scores.
    """
    for item in data:
        target = item['structure']
        spans = {int(k): v for k, v in item['spans'].items()}
        for pos, stack in target.items():
            if not stack or any(b.endswith('l') for b in stack):
                continue
            for depth in range(len(stack) + 1):
                yield item['id'], target, spans, pos, stack[:depth], 'clean'
                for wrong in range(depth):
                    corrupt = stack[:depth]
                    corrupt[wrong] = next(b for b in ['rs', 'bs', 'gs'] if b != stack[wrong])
                    label = ('top' if wrong == depth - 1 else 'buried') + (
                        '_full' if depth == len(stack) else '_short')
                    yield item['id'], target, spans, pos, corrupt, label


# Accumulated by ProgressChecks, printed by tearDownModule.
REPORT = {}


def rollout(g, fn=enumerate_correct_actions, limit=30):
    seen, trace = set(), []
    for _ in range(limit):
        if exact(g):
            return True, trace, 'exact'
        key = signature(g)
        if key in seen:
            return False, trace, 'cycle'
        seen.add(key)
        entries = ok_entries(g, fn)
        if not entries:
            return False, trace, 'stuck'
        move = entries[0]['move']
        trace.append((move['action'], move['block'], move['layer']))
        execute(g, move)
    return False, trace, 'limit'


def candidate_function():
    """Add a mismatch predicate, guard placement, and bound the mismatch loop.

    No persistent latch is needed.

    Keep excess removal as-is. Permit placement only on a matching prefix.
    Existing mismatch removal code now runs for short corrupt stacks too.
    Fail loudly if the source no longer matches the version these tests target.
    """
    source = textwrap.dedent(inspect.getsource(enumerate_correct_actions))
    edits = [
        ('if current_depth < target_depth:',
         'has_mismatch = any(a != b for a, b in zip(current_stack, target_stack))\n'
         '        if current_depth < target_depth and not has_mismatch:'),
        ('for layer_idx in range(current_depth):',
         'for layer_idx in range(min(current_depth, target_depth)):'),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError(f'Candidate patch requires exactly one occurrence: {old}')
        source = source.replace(old, new)
    namespace = dict(vars(oracle_module))
    exec(compile(source, '<isolated-prefix-fix>', 'exec'), namespace)
    return namespace['enumerate_correct_actions']


class CurrentBehavior(unittest.TestCase):
    def test_four_pathways(self):
        cases = [
            (['gs'], ['gs', 'os', 'bs'], 'target_place', 'place', 'os', 1),
            (['gs', 'os', 'bs'], ['gs', 'os'], 'excess_remove', 'remove', 'bs', 2),
            (['gs', 'os', 'rs'], ['gs', 'os', 'bs'], 'wrong_block_remove', 'remove', 'rs', 2),
            (['gs', 'rs', 'bs'], ['gs', 'os', 'bs'], 'expose_buried_wrong', 'remove', 'bs', 2),
        ]
        for current, target, source, action, block, layer in cases:
            with self.subTest(source=source):
                g = game(board({P: target}), board({P: current}))
                entries = ok_entries(g)
                self.assertEqual(len(entries), 1)
                e = entries[0]
                self.assertEqual(e['source'], source)
                self.assertEqual((e['move']['action'], e['move']['block'], e['move']['layer']),
                                 (action, block, layer))
                execute(g, e['move'])

    def test_short_corrupt_stack_currently_gets_place(self):
        g = game(board({P: ['gs', 'os', 'bs']}), board({P: ['gs', 'rs']}))
        entries = ok_entries(g)
        self.assertEqual([(e['source'], e['move']['action'], e['move']['block'])
                          for e in entries], [('target_place', 'place', 'bs')])
        result = execute(g, entries[0]['move'])
        self.assertTrue(result[2])  # Environment's LOCAL structurePlacement is true.
        self.assertEqual(g.current_structure[P], ['gs', 'rs', 'bs'])

    def test_buried_wrong_remove_replace_cycle(self):
        g = game(board({P: ['gs', 'os', 'bs']}), board({P: ['gs', 'rs', 'bs']}))
        success, trace, reason = rollout(g)
        self.assertFalse(success)
        self.assertEqual(reason, 'cycle')
        self.assertEqual(trace, [('remove', 'bs', 2), ('place', 'bs', 2)])

    def test_oracle_flag_and_environment_score_differ_for_clearing(self):
        g = game(board({P: ['gs', 'os', 'bs']}), board({P: ['gs', 'rs', 'bs']}))
        before = g.progress_tracker.calculate_progress(g.current_structure)['overall_progress']
        e = ok_entries(g)[0]
        result = execute(g, e['move'])
        after = g.progress_tracker.calculate_progress(g.current_structure)['overall_progress']
        self.assertTrue(e['structurePlacement'])
        self.assertFalse(result[2])
        self.assertLess(after, before)

    def test_exact_target_has_no_moves(self):
        target = board({P: ['gs', 'os']})
        self.assertEqual(ok_entries(game(target, target)), [])

    def test_enumeration_does_not_mutate_board_or_target(self):
        g = game(board({P: ['gs', 'os', 'bs']}), board({P: ['gs', 'rs', 'bs']}))
        before = copy.deepcopy((g.current_structure, g.target_structure,
                                g.current_spans, g.target_spans, g.progress_tracker.progress_history))
        enumerate_correct_actions(g)
        self.assertEqual(before, (g.current_structure, g.target_structure,
                                  g.current_spans, g.target_spans, g.progress_tracker.progress_history))

    def test_large_placement_dedup_and_atomic_removal(self):
        target = board({P: ['gl'], Q: ['gl']})
        g = game(target, target_spans={0: [(P, Q)]})
        entries = ok_entries(g)
        self.assertEqual(len(entries), 1)
        execute(g, entries[0]['move'])
        self.assertTrue(exact(g))
        g.target_structure = board()
        removals = ok_entries(g)
        self.assertEqual(len(removals), 1)
        execute(g, removals[0]['move'])
        self.assertEqual(g.current_structure, board())
        self.assertEqual(canonical_spans(g.current_spans), {})

    def test_missing_span_is_not_an_ok_candidate(self):
        g = game(board({P: ['gl'], Q: ['gl']}))
        self.assertEqual(ok_entries(g), [])
        self.assertTrue(all(e['flag'] == 'missing_span_info'
                            for e in enumerate_correct_actions(g)))

    def test_unequal_support_heights_reject_large_placement(self):
        g = game(board({P: ['gs', 'gl'], Q: ['gs', 'gl']}),
                 board({P: ['gs']}), target_spans={1: [(P, Q)]})
        entries = enumerate_correct_actions(g)
        self.assertTrue(any(e['flag'] == 'sim_failed' and e['move']['block'] == 'gl'
                            for e in entries))
        self.assertTrue(all(e['move']['block'] != 'gl' for e in ok_entries(g)))

    def test_blocked_large_removal_fails_without_mutation(self):
        g = game(board(), board({P: ['gl'], Q: ['gl', 'bs']}),
                 current_spans={0: [(P, Q)]})
        move = dict(action='remove', block='gl', position=P, layer=0, span_to=Q)
        before = copy.deepcopy((g.current_structure, g.current_spans))
        self.assertTrue(g._validate_move(copy.deepcopy(move))[0])  # validator is incomplete
        result = g.execute_move(copy.deepcopy(move))
        self.assertFalse(result[0])  # execution DOES reject it safely
        self.assertEqual(before, (g.current_structure, g.current_spans))
        self.assertFalse(any(e['move']['block'] == 'gl' for e in ok_entries(g)))

    def test_default_invisible_rule_differs_from_original_suite(self):
        target = board({R: ['gl'], S: ['gl']})
        allowed = game(target, target_spans={0: [(R, S)]})
        production_default = game(target, target_spans={0: [(R, S)]}, invisible_cells=None)
        self.assertEqual(len(ok_entries(allowed)), 1)
        self.assertEqual(ok_entries(production_default), [])

    def test_sampling_count_membership_reproducibility_and_no_rng(self):
        g = game(board({p: ['gs'] for p in COORDS}))
        all_moves = [e['move'] for e in ok_entries(g)]
        for n in (0, 1, 5, 100):
            a = get_oracle_moves(g, n=n, rng=random.Random(42))
            b = get_oracle_moves(g, n=n, rng=random.Random(42))
            self.assertEqual(a, b)
            self.assertEqual(len(a), min(n, len(all_moves)))
            self.assertTrue(all(m in all_moves for m in a))
            self.assertEqual(len({repr(m) for m in a}), len(a))
        self.assertEqual(get_oracle_moves(g, n=5), all_moves[:5])


class CandidateDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed = staticmethod(candidate_function())

    def test_short_wrong_top_delta(self):
        g = game(board({P: ['gs', 'os', 'bs']}), board({P: ['gs', 'rs']}))
        self.assertEqual(ok_entries(g)[0]['move']['action'], 'place')
        self.assertEqual(ok_entries(g, self.fixed)[0]['move']['action'], 'remove')
        success, trace, _ = rollout(g, self.fixed)
        self.assertTrue(success)
        self.assertEqual(trace, [('remove', 'rs', 1), ('place', 'os', 1), ('place', 'bs', 2)])

    def test_buried_delta(self):
        g = game(board({P: ['gs', 'os', 'bs']}), board({P: ['gs', 'rs', 'bs']}))
        success, trace, _ = rollout(g, self.fixed)
        self.assertTrue(success)
        self.assertEqual(trace, [('remove', 'bs', 2), ('remove', 'rs', 1),
                                 ('place', 'os', 1), ('place', 'bs', 2)])

    def test_every_single_wrong_layer_at_every_nonempty_depth(self):
        target = ['gs', 'os', 'bs']
        current_successes = fixed_successes = 0
        for depth in (1, 2, 3):
            for layer in range(depth):
                with self.subTest(depth=depth, wrong_layer=layer):
                    stack = target[:depth]
                    stack[layer] = 'rs'
                    g = game(board({P: target}), board({P: stack}))
                    current_successes += rollout(copy.deepcopy(g))[0]
                    fixed_successes += rollout(copy.deepcopy(g), self.fixed)[0]
        self.assertEqual(current_successes, 1)
        self.assertEqual(fixed_successes, 6)

    def test_clean_prefixes_and_excess_keep_identical_candidates(self):
        for depth in (0, 1, 2, 3):
            g = game(board({P: ['gs', 'os', 'bs']}),
                     board({P: ['gs', 'os', 'bs'][:depth]}))
            self.assertEqual(enumerate_correct_actions(g), self.fixed(g))
        g = game(board({P: ['gs']}), board({P: ['gs', 'rs', 'bs']}))
        self.assertEqual(enumerate_correct_actions(g), self.fixed(g))
        self.assertTrue(rollout(g, self.fixed)[0])


class KnownGaps(unittest.TestCase):
    """Desired invariants the enumerator does not yet satisfy. Remove expectedFailure once fixed."""
    @unittest.expectedFailure
    def test_short_corrupt_stack_should_offer_removal(self):
        g = game(board({P: ['gs', 'os', 'bs']}), board({P: ['gs', 'rs']}))
        self.assertTrue(any(e['move']['action'] == 'remove' for e in ok_entries(g)))

    @unittest.expectedFailure
    def test_current_oracle_should_repair_buried_wrong(self):
        g = game(board({P: ['gs', 'os', 'bs']}), board({P: ['gs', 'rs', 'bs']}))
        self.assertTrue(rollout(g)[0])

    @unittest.expectedFailure
    def test_stray_blocks_outside_sparse_target_should_be_removed(self):
        g = game({P: ['gs']}, board({P: ['gs'], Q: ['rs']}))
        self.assertTrue(any(e['move']['action'] == 'remove' and e['move']['position'] == Q
                            for e in ok_entries(g)))

    @unittest.expectedFailure
    def test_matching_codes_wrong_span_topology_should_offer_repair(self):
        target = board({p: ['gl'] for p in (P, Q, R, S)})
        g = game(target, target, target_spans={0: [(P, R), (Q, S)]},
                 current_spans={0: [(P, Q), (R, S)]})
        self.assertFalse(exact(g))
        self.assertTrue(ok_entries(g))

    @unittest.expectedFailure
    def test_partial_wall_must_not_alias_target(self):
        target = board({P: ['gs']})
        g = game(target, partComplete=True, partType='D1Wall')
        before = copy.deepcopy(g.target_structure)
        execute(g, dict(action='remove', block='gs', position=P, layer=0, span_to=None))
        self.assertEqual(g.target_structure, before)

    @unittest.expectedFailure
    def test_partial_first_layer_must_record_large_spans(self):
        g = game(board({P: ['gl'], Q: ['gl']}), target_spans={0: [(P, Q)]},
                 partComplete=True, partType='firstLayer')
        self.assertEqual(canonical_spans(g.current_spans), {0: [(P, Q)]})

    @unittest.expectedFailure
    def test_95_percent_completion_does_not_guarantee_exact_target(self):
        # 27 target cells/layers; remove one repeated gs. Set-based metrics stay perfect.
        target = board({p: ['gs', 'gs', 'gs'] for p in COORDS})
        g = game(target, target)
        execute(g, dict(action='remove', block='gs', position=P, layer=2, span_to=None))
        self.assertFalse(exact(g))
        self.assertFalse(g.is_complete())

    @unittest.expectedFailure
    def test_local_prefix_fix_does_not_resolve_cross_cell_blockers(self):
        target = board({P: ['gs', 'gl'], Q: ['gs', 'gl', 'bs']})
        g = game(target, board({P: ['rs', 'gl'], Q: ['gs', 'gl', 'bs']}),
                 target_spans={1: [(P, Q)]}, current_spans={1: [(P, Q)]})
        # Need to remove correct bs at Q before removing the shared gl and wrong rs.
        self.assertTrue(rollout(g, candidate_function())[0])



DATASET = Path(os.environ.get('CRAFT_DATASET', 'data/structures_dataset_20.json'))


@unittest.skipUnless(DATASET.exists(), 'Set CRAFT_DATASET or supply data/structures_dataset_20.json')
class DatasetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DATASET.read_text())
        cls.fixed = staticmethod(candidate_function())

    def test_exact_clean_builds_with_original_suite_configuration(self):
        for item in self.data:
            with self.subTest(structure=item['id']):
                spans = {int(k): v for k, v in item['spans'].items()}
                g = game(item['structure'], target_spans=spans)
                self.assertTrue(rollout(g, limit=60)[0])

    def test_default_configuration_exposes_forbidden_target_spans(self):
        completed = forbidden_targets = 0
        for item in self.data:
            spans = {int(k): v for k, v in item['spans'].items()}
            g = game(item['structure'], target_spans=spans, invisible_cells=None)
            forbidden = any(a in g.invisible_cells or b in g.invisible_cells
                            for pairs in spans.values() for a, b in pairs)
            success = rollout(g, limit=60)[0]
            forbidden_targets += forbidden
            completed += success
            if forbidden:
                self.assertFalse(success)
            else:
                self.assertTrue(success)
        print(f'Default rules: exact {completed}/{len(self.data)}; '
              f'targets with forbidden spans: {forbidden_targets}')

    def test_corruption_delta_and_clean_prefix_equivalence(self):
        stats = defaultdict(lambda: [0, 0, 0])
        clean = 0
        for item in self.data:
            target = item['structure']
            spans = {int(k): v for k, v in item['spans'].items()}
            for pos, stack in target.items():
                if not stack or any(b.endswith('l') for b in stack):
                    continue
                for depth in range(len(stack) + 1):
                    current = copy.deepcopy(target)
                    current[pos] = stack[:depth]
                    g = game(target, current, target_spans=spans, current_spans=spans)
                    self.assertEqual(enumerate_correct_actions(g), self.fixed(g))
                    clean += 1
                    for wrong in range(depth):
                        current = copy.deepcopy(target)
                        current[pos] = stack[:depth]
                        current[pos][wrong] = next(b for b in ['rs', 'bs', 'gs']
                                                  if b != stack[wrong])
                        g = game(target, current, target_spans=spans, current_spans=spans)
                        before = rollout(copy.deepcopy(g))[0]
                        after = rollout(copy.deepcopy(g), self.fixed)[0]
                        label = ('top' if wrong == depth - 1 else 'buried') + (
                            '_full' if depth == len(stack) else '_short')
                        stats[label][0] += 1
                        stats[label][1] += before
                        stats[label][2] += after
                        with self.subTest(structure=item['id'], pos=pos, depth=depth, wrong=wrong):
                            self.assertEqual(before, wrong == depth - 1 and depth == len(stack))
                            self.assertTrue(after)
        self.assertGreater(sum(row[0] for row in stats.values()), 0)
        print('Injection counts [cases, current exact repairs, candidate exact repairs]:', dict(stats))
        print(f'Identical candidate lists on {clean} clean-prefix states')


@unittest.skipUnless(DATASET.exists(), 'Set CRAFT_DATASET or supply data/structures_dataset_20.json')
class ProgressChecks(unittest.TestCase):
    """Does a suggested move advance the environment's own progress metric?

    The paper states oracle moves "reflect locally verified progress from s_t
    rather than a globally optimal trajectory toward T". These tests measure
    both halves of that claim: the per-move delta (local) and the net delta
    across a full rollout (global).
    """
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DATASET.read_text())
        cls.fixed = staticmethod(candidate_function())

    def test_per_move_progress_delta_by_source(self):
        """Characterize Δprogress for every ok candidate, grouped by branch."""
        stats = defaultdict(lambda: {'n': 0, 'up': 0, 'flat': 0, 'down': 0,
                                     'sum_dp': 0.0, 'sum_db': 0})
        for _, target, spans, pos, stack, _ in injection_cases(self.data):
            current = copy.deepcopy(target)
            current[pos] = stack
            g = game(target, current, target_spans=spans, current_spans=spans)
            before_p, before_b = progress(g), blocks_correct(g)
            for entry in ok_entries(g):
                probe = game(target, current, target_spans=spans, current_spans=spans)
                if not probe.execute_move(copy.deepcopy(entry['move']))[0]:
                    continue
                dp = progress(probe) - before_p
                row = stats[entry['source']]
                row['n'] += 1
                row['sum_dp'] += dp
                row['sum_db'] += blocks_correct(probe) - before_b
                row['up' if dp > 1e-9 else 'down' if dp < -1e-9 else 'flat'] += 1
        REPORT['per_move'] = {k: dict(v) for k, v in stats.items()}

        self.assertIn('target_place', stats)
        # The primary branch must never lose ground on the environment's metric.
        self.assertEqual(stats['target_place']['down'], 0)
        self.assertEqual(stats['target_place']['flat'], 0)
        # Clearing a correct block to reach a buried wrong one necessarily does.
        if stats['expose_buried_wrong']['n']:
            self.assertGreater(stats['expose_buried_wrong']['down'], 0)

    def test_net_progress_over_rollout_current_vs_candidate(self):
        """Per-move deltas can mislead; what matters is where the rollout ends."""
        totals = defaultdict(lambda: {'n': 0,
                                      'cur_exact': 0, 'fix_exact': 0,
                                      'cur_net': 0.0, 'fix_net': 0.0,
                                      'cur_cycle': 0, 'cur_stuck': 0})
        for _, target, spans, pos, stack, label in injection_cases(self.data):
            if label == 'clean':
                continue
            current = copy.deepcopy(target)
            current[pos] = stack
            row = totals[label]
            row['n'] += 1
            for tag, fn in (('cur', enumerate_correct_actions), ('fix', self.fixed)):
                g = game(target, current, target_spans=spans, current_spans=spans)
                start = progress(g)
                success, _, reason = rollout(g, fn)
                row[f'{tag}_net'] += progress(g) - start
                row[f'{tag}_exact'] += bool(success)
                if tag == 'cur' and reason in ('cycle', 'stuck'):
                    row[f'cur_{reason}'] += 1
        REPORT['rollout'] = {k: dict(v) for k, v in totals.items()}

        for label, row in totals.items():
            with self.subTest(label=label):
                # The candidate fix must finish every injected corruption.
                self.assertEqual(row['fix_exact'], row['n'])
                # Current logic only recovers when the wrong block is already on top
                # of a full-depth stack — every other class ends non-exact.
                if label != 'top_full':
                    self.assertEqual(row['cur_exact'], 0)


def tearDownModule():
    if not REPORT:
        return
    line = '-' * 78
    print(f'\n{line}\nPROGRESS REPORT — environment overall_progress, measured on dataset states')
    print(line)

    per_move = REPORT.get('per_move')
    if per_move:
        print('\nPer-move Δprogress by enumeration branch (all ok candidates):\n')
        print(f"  {'branch':<22}{'n':>6}{'Δ>0':>7}{'Δ=0':>7}{'Δ<0':>7}"
              f"{'mean Δprog':>13}{'mean Δblocks':>14}")
        total = defaultdict(int)
        for source in sorted(per_move):
            r = per_move[source]
            for k in ('n', 'up', 'flat', 'down'):
                total[k] += r[k]
            print(f"  {source:<22}{r['n']:>6}{r['up']:>7}{r['flat']:>7}{r['down']:>7}"
                  f"{r['sum_dp'] / r['n']:>+13.4f}{r['sum_db'] / r['n']:>+14.2f}")
        n = total['n'] or 1
        print(f"\n  {'ALL':<22}{total['n']:>6}{total['up']:>7}{total['flat']:>7}{total['down']:>7}")
        print(f"  advancing {total['up'] / n:.1%} | flat {total['flat'] / n:.1%} "
              f"| regressive {total['down'] / n:.1%}")
        uncovered = {'target_place', 'excess_remove',
                     'wrong_block_remove', 'expose_buried_wrong'} - set(per_move)
        if uncovered:
            print(f"  NOT EXERCISED here: {', '.join(sorted(uncovered))} — injection_cases only"
                  f"\n  truncates or corrupts a stack, never over-builds it. Branch correctness"
                  f"\n  is covered by CurrentBehavior.test_four_pathways; its Δprogress is not.")

    rollout_stats = REPORT.get('rollout')
    if rollout_stats:
        print('\nNet Δprogress across a full rollout from each injected corruption:\n')
        print(f"  {'corruption':<16}{'n':>5}{'current exact':>15}{'cand exact':>12}"
              f"{'cur net Δ':>12}{'cand net Δ':>12}{'cur cycles':>12}")
        agg = defaultdict(float)
        for label in sorted(rollout_stats):
            r = rollout_stats[label]
            for k in ('n', 'cur_exact', 'fix_exact', 'cur_net', 'fix_net', 'cur_cycle'):
                agg[k] += r[k]
            print(f"  {label:<16}{r['n']:>5}{r['cur_exact']:>15}{r['fix_exact']:>12}"
                  f"{r['cur_net']:>+12.3f}{r['fix_net']:>+12.3f}{r['cur_cycle']:>12}")
        n = int(agg['n']) or 1
        print(f"  {'OVERALL':<16}{n:>5}{int(agg['cur_exact']):>15}{int(agg['fix_exact']):>12}"
              f"{agg['cur_net']:>+12.3f}{agg['fix_net']:>+12.3f}{int(agg['cur_cycle']):>12}")
        print(f"\n  exact-repair rate: current {agg['cur_exact'] / n:.1%} "
              f"-> candidate {agg['fix_exact'] / n:.1%} "
              f"({(agg['fix_exact'] - agg['cur_exact']) / n:+.1%})")
    print(line)


if __name__ == '__main__':
    unittest.main(verbosity=2)
