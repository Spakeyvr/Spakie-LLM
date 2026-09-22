import re
import unittest
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts.probe_base_learning import answer, build_tasks, main, pair_key


class BaseLearningProbeTests(unittest.TestCase):
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


if __name__=='__main__':
    unittest.main()
