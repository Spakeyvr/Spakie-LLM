import re
import json
import copy
import unittest
import numpy as np
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts.probe_base_learning import (answer, build_tasks, main, pair_key, probe_learning_rate,
                                        verified_document_starts, snap_to_document_start,
                                        generation_task_fields, validate_probe_initialization)


class BaseLearningProbeTests(unittest.TestCase):
    def test_probe_context_cli_preserves_default_and_rejects_invalid_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            common=['probe','run','--assets',directory,'--data',directory,
                    '--output',directory+'-output','--arm','curriculum']
            for flags,expected in (([],256),(['--sequence-length','1024'],1024),(['--sequence-length','2048'],2048)):
                with patch('sys.argv',common+flags),patch('scripts.probe_base_learning.run',return_value=0) as run:
                    self.assertEqual(main(),0)
                    self.assertEqual(run.call_args.args[0].sequence_length,expected)
            for flags in (['--sequence-length','255'],['--sequence-length','4096'],
                          ['--arm','sanity','--sequence-length','1024']):
                with patch('sys.argv',common+flags),patch('scripts.probe_base_learning.run') as run,redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:main()
                    self.assertEqual(error.exception.code,2)
                    run.assert_not_called()

    def test_replay_anchor_cli_accepts_strong_probe_and_rejects_invalid_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            common=['probe','run','--assets',directory,'--data',directory,
                    '--output',directory+'-output','--arm','curriculum',
                    '--replay-loss','kl','--replay-kl-weight']
            with patch('sys.argv',common+['20']),patch('scripts.probe_base_learning.run',return_value=0) as run:
                self.assertEqual(main(),0)
                self.assertEqual(run.call_args.args[0].replay_kl_weight,20.)
            for value in ('-1','0','20.001','nan','inf'):
                with patch('sys.argv',common+[value]),patch('scripts.probe_base_learning.run') as run,redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:main()
                    self.assertEqual(error.exception.code,2)
                    run.assert_not_called()

    def test_staged_initialization_rejects_wrong_lineage_and_contracts(self):
        parent={'config':{'width':8},'tokenizer':{'sha':'frozen'},'step':100,'tokens_processed':1000}
        initial={**copy.deepcopy(parent),'stage':'pretrain','step':102,'tokens_processed':3048,
                 'experiment':{'parent_sha256':'root','inference_only':True,'status':'complete'}}
        validate_probe_initialization(parent,initial,'root')
        for field,value in [('stage','sft'),('config',{'width':16}),('tokenizer',{}),
                            ('step',99),('step',True),('tokens_processed',999)]:
            invalid=copy.deepcopy(initial);invalid[field]=value
            with self.assertRaises(ValueError):validate_probe_initialization(parent,invalid,'root')
        for field,value in [('parent_sha256','other'),('inference_only',False),
                            ('status','interrupted'),('blend',{'fraction':.5})]:
            invalid=copy.deepcopy(initial);invalid['experiment'][field]=value
            with self.assertRaises(ValueError):validate_probe_initialization(parent,invalid,'root')

    def test_staged_initialization_cli_rejects_ambiguous_or_nontraining_use(self):
        with tempfile.TemporaryDirectory() as directory:
            common=['probe','run','--assets',directory,'--data',directory,
                    '--output',directory+'-output','--initialize-from',directory+'/initial.safetensors']
            for flags in ([],['--blend-from',directory+'/blend.safetensors']):
                with patch('sys.argv',common+flags),redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:main()
                self.assertEqual(error.exception.code,2)

    def test_task_metadata_cannot_collide_with_generation_log_fields(self):
        row = {'id':'numeric', 'category':'selection', 'kind':'curriculum',
               'phase':'fake', 'passed':True, 'output':'fake', 'seed':7,
               'task_metadata':{'original':True}}
        fields = generation_task_fields(row)
        def emit(kind, *, phase, output, passed, seed, **remaining):
            return {'kind':kind,'phase':phase,'output':output,'passed':passed,
                    'seed':seed,**remaining}
        record = emit('generation',phase='after',**fields,output='actual',passed=False,seed=42)
        self.assertEqual(record['kind'],'generation')
        self.assertEqual(record['output'],'actual')
        self.assertFalse(record['passed'])
        self.assertEqual(record['task_metadata']['kind'],'curriculum')
        self.assertEqual(record['task_metadata']['task_metadata'],{'original':True})
        self.assertEqual(row['output'],'fake')
        self.assertEqual(generation_task_fields({'id':'clean'}),{'id':'clean'})

    def test_document_windows_validate_packing_and_preserve_complete_first_document(self):
        class Tokenizer:
            def encode(self, text, add_eos=False):
                return [int(c) for c in text] + ([0] if add_eos else [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'curriculum.jsonl'
            path.write_text(''.join(json.dumps({'text':text})+'\n' for text in ('12','345','67')))
            tokens = np.array([1,2,0,3,4,5,0,6,7,0])
            starts = verified_document_starts(path,Tokenizer(),tokens,window=4)
            self.assertEqual(starts.tolist(),[0,3,7])
            self.assertEqual([snap_to_document_start(i,starts) for i in range(7)], [0,0,0,3,3,3,3])
            for offset in range(len(tokens)-4):
                aligned = snap_to_document_start(offset,starts)
                self.assertEqual(len(tokens[aligned:aligned+4]),4)
                self.assertIn(0,tokens[aligned:aligned+4])
            for broken in (tokens[:-1],np.r_[tokens,8],tokens.reshape(2,5),tokens+1):
                with self.assertRaises(ValueError):
                    verified_document_starts(path,Tokenizer(),broken,window=4)
            with self.assertRaises(ValueError):
                verified_document_starts(path,Tokenizer(),tokens,window=3)
            with self.assertRaises(KeyboardInterrupt):
                verified_document_starts(path,Tokenizer(),tokens,window=4,stopped=lambda: True)

    def test_probe_schedule_preserves_constant_and_anneals_to_declared_floor(self):
        peak = 1e-4
        self.assertEqual([probe_learning_rate(i,5,peak) for i in range(5)], [peak]*5)
        rates = [probe_learning_rate(i,5,peak,'cosine') for i in range(5)]
        self.assertAlmostEqual(rates[0], peak)
        self.assertAlmostEqual(rates[2], peak*.55)
        self.assertAlmostEqual(rates[-1], peak*.1)
        self.assertTrue(all(a>b for a,b in zip(rates,rates[1:])))
        self.assertEqual(probe_learning_rate(0,1,peak,'cosine'),peak)
        for step,total,rate,schedule in [(-1,5,peak,'constant'),(5,5,peak,'cosine'),
                                         (0,0,peak,'constant'),(0,5,float('nan'),'constant'),
                                         (0,5,0,'constant'),(0,5,peak,'unknown')]:
            with self.assertRaises(ValueError):
                probe_learning_rate(step,total,rate,schedule)

    def test_capital_city_descriptor_accepts_only_the_immediate_correct_city(self):
        from scripts._base_probe_cases import score_completion
        row={'category':'facts','prompt':'The capital of Switzerland is','expected_regex':r'Bern\b'}
        for text in ['Bern.', 'the city of Bern.', 'the capital city of Bern.',
                     'in the city of Bern, which is located', 'in Bern.']:
            self.assertTrue(score_completion(text,row),text)
        for text in ['the city of Geneva. Bern is another city.', 'Not Bern.',
                     'the capital city of the country. Later: Bern.',
                     'in the city of Geneva. Bern.', 'in the south. Bern.']:
            self.assertFalse(score_completion(text,row),text)

    def test_legacy_base_detection_requires_corpus_contract_and_rejects_sft(self):
        from scripts._base_probe_cases import is_base_checkpoint
        legacy = {'config':{'vocab_size':100}, 'sampler':{'offset':0},
                  'tokens_processed':1024, 'processed_data_manifest_sha256':'a'*64}
        self.assertTrue(is_base_checkpoint(legacy, 'pretrain_interrupt.safetensors'))
        self.assertFalse(is_base_checkpoint(legacy, 'sft_interrupt.safetensors'))
        self.assertFalse(is_base_checkpoint({**legacy,'stage':'sft'}, 'pretrain_interrupt.safetensors'))
        self.assertFalse(is_base_checkpoint({**legacy,'sampler':{}}, 'pretrain_interrupt.safetensors'))
        self.assertFalse(is_base_checkpoint({'config':{'vocab_size':100}}, 'pretrain_interrupt.safetensors'))
        self.assertTrue(is_base_checkpoint({'stage':'pretrain'}, 'base_probe.safetensors'))

    def test_legacy_numeric_oracles_accept_sentence_periods_but_not_wrong_decimals(self):
        from scripts._base_probe_cases import score_completion
        row = {'category':'arithmetic_words','expected_regex':r'76(?:\.0+)?(?![\d.])'}
        for text in ('76.', '76.\nAnother problem.', '76', '76.00 apples'):
            self.assertTrue(score_completion(text,row),text)
        for text in ('760.', '76.5', '76.5x', '76e2', '-76.', 'Wrong. 76.'):
            self.assertFalse(score_completion(text,row),text)

    def test_rescoring_preserves_original_results_and_separates_phases(self):
        from scripts.rescore_base_probe import rescore
        row = {'kind':'generation','id':'sum','category':'arithmetic_words',
               'prompt':'53 plus 23 equals','expected_regex':r'76(?![\d.])',
               'output':'76.\nNext example.','passed':False}
        result = rescore([{**row,'phase':'before'},{**row,'phase':'after'}])
        for phase in ('before','after'):
            self.assertEqual(result['metrics'][phase]['arithmetic_words'],
                             {'passed':1,'original_passed':0,'total':1})
        self.assertEqual(len(result['changed_records']),2)
        self.assertFalse(row['passed'])

    def test_factual_descriptor_does_not_hide_wrong_or_later_answers(self):
        from scripts._base_probe_cases import score_completion
        row = {'id':'fact_hamlet','category':'facts','expected_regex':r'(?:William )?Shakespeare\b'}
        self.assertTrue(score_completion('the English playwright William Shakespeare.',row))
        self.assertTrue(score_completion('the English playwright and poet, William Shakespeare.',row))
        for text in ('George R. R. Martin.', 'the English playwright George R. R. Martin.',
                     'Not Shakespeare.', 'George Martin, not William Shakespeare.'):
            self.assertFalse(score_completion(text,row),text)

    @classmethod
    def setUpClass(cls):
        cls.curriculum, cls.tasks = build_tasks()

    def test_arithmetic_holdouts_exclude_commuted_training_pairs(self):
        held = {tuple(row['pair']) for row in self.tasks if 'pair' in row}
        trained = {tuple(pair) for row in self.curriculum for pair in row.get('pairs', [])}
        self.assertTrue(held)
        self.assertTrue(trained)
        self.assertTrue(held.isdisjoint(trained))
        self.assertEqual(pair_key('+', 2, 9), pair_key('+', 9, 2))
        self.assertNotEqual(pair_key('-', 2, 9), pair_key('-', 9, 2))

    def test_every_generated_equation_has_the_correct_target(self):
        count = 0
        for row in self.curriculum:
            for a,op,b,c in re.findall(r'(\d+) ([+*/-]) (\d+) = (\d+)', row['text']):
                self.assertEqual(answer(op,int(a),int(b)),int(c))
                count += 1
        self.assertGreater(count, 10000)

    def test_probe_strings_are_not_training_literals(self):
        trained = {r['string'] for r in self.curriculum if 'string' in r}
        for row in self.tasks:
            if row['category']=='code_length':
                text = re.search(r'len\("(.*)"\)', row['prompt'])[1]
                self.assertNotIn(text, trained)

    def test_worked_examples_exclude_independent_readiness_operands(self):
        from scripts.eval_base_readiness import cases
        curriculum, tasks = build_tasks(worked_examples=True)
        trained = {tuple(p) for row in curriculum for p in row.get('pairs', [])}
        held = {tuple(r['pair']) for r in tasks if 'pair' in r}
        held.update(pair_key('+', *r['operands']) for split in ('dev','test')
                    for r in cases(split) if 'operands' in r)
        self.assertTrue(trained.isdisjoint(held))
        worked = [r for r in curriculum if 'Calculation:' in r['text']]
        self.assertGreater(len(worked), 1000)
        for row in worked:
            for a,op,b,c in re.findall(r'Calculation: (\d+) ([+*/-]) (\d+) = (\d+)', row['text']):
                self.assertEqual(answer(op,int(a),int(b)), int(c))

    def test_hard_limits_reject_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output'
            for steps,seconds in [(257,240),(256,241),(0,1),(1,0)]:
                argv=['probe','prepare','--assets',str(Path(directory)/'assets'),
                      '--output',str(output),'--steps',str(steps),'--seconds',str(seconds)]
                with patch('sys.argv',argv),redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:main()
                self.assertEqual(error.exception.code,2)
                self.assertFalse(output.exists())

    def test_outputs_cannot_be_written_inside_original_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'original_assets'/'new_output'
            argv=['probe','prepare','--assets',str(output.parent),'--output',str(output)]
            with patch('sys.argv',argv),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:main()
            self.assertEqual(error.exception.code,2)
            self.assertFalse(output.exists())

    def test_extended_pilot_still_has_hard_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'output'
            for steps, seconds in [(2049,600), (2048,601)]:
                argv = ['probe','run','--assets',str(Path(directory)/'assets'),
                        '--data',str(Path(directory)/'data'),'--output',str(output),
                        '--extended-pilot','--steps',str(steps),'--seconds',str(seconds)]
                with patch('sys.argv',argv), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main()
                self.assertEqual(error.exception.code,2)
            self.assertFalse(output.exists())

    def test_token_pilot_retains_time_limit_and_rejects_ambiguous_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'output'
            common = ['probe','run','--assets',str(Path(directory)/'assets'),
                      '--data',str(Path(directory)/'data'),'--output',str(output)]
            for flags in [ ['--token-pilot','--steps','4097'],
                           ['--token-pilot','--seconds','601'],
                           ['--token-pilot','--extended-pilot'] ]:
                with patch('sys.argv',common+flags), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main()
                self.assertEqual(error.exception.code,2)
                self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
