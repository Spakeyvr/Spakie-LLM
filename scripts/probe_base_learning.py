"""Small BASE-only learning probes, with hard update/time limits and isolated outputs.

Prepare frozen arithmetic/code tasks, then compare corpus replay and a verified
completion curriculum. This is ordinary next-token learning, not chat SFT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizer.train_tokenizer import SpakieTokenizer


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def pair_key(op, a, b):
    return (op, min(a, b), max(a, b)) if op in ('+', '*') else (op, a, b)


def verified_document_starts(path, tokenizer, tokens, *, window=257, stopped=lambda: False):
    """Validate JSONL against every packed token before using its boundaries."""
    if tokens.ndim != 1 or len(tokens) <= window:
        raise ValueError('Curriculum must be a one-dimensional array longer than a window')
    starts = []
    offset = 0
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if stopped():
                raise KeyboardInterrupt
            row = json.loads(line)
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                raise ValueError(f'Invalid curriculum text at line {line_number}')
            ids = tokenizer.encode(row['text'], add_eos=True)
            if not 1 <= len(ids) <= window:
                raise ValueError(f'Document at line {line_number} does not fit the training window')
            if not np.array_equal(tokens[offset:offset+len(ids)], ids):
                raise ValueError(f'Curriculum JSONL/token mismatch at line {line_number}')
            starts.append(offset)
            offset += len(ids)
    if offset != len(tokens):
        raise ValueError('Curriculum JSONL does not cover the complete token array')
    return np.asarray(starts, dtype=np.int64)


def snap_to_document_start(offset, starts):
    """Move a sampled token position backward without drawing additional RNG."""
    if not len(starts) or starts[0] != 0 or offset < 0:
        raise ValueError('Document boundaries must start at zero; offset must be nonnegative')
    return int(starts[np.searchsorted(starts, offset, side='right')-1])


def generation_task_fields(row):
    """Keep task metadata from colliding with authoritative evaluation fields."""
    reserved = {'kind', 'phase', 'output', 'output_ids', 'passed', 'temperature',
                'seed', 'repeat_4gram_fraction', 'task_metadata'}
    fields = {key:value for key,value in row.items() if key not in reserved}
    metadata = {key:value for key,value in row.items() if key in reserved}
    if metadata:
        fields['task_metadata'] = metadata
    return fields


def validate_probe_initialization(parent, initial, parent_sha):
    """Accept only completed, unblended BASE probe weights from this frozen root."""
    experiment = initial.get('experiment', {})
    if (initial.get('stage') != 'pretrain' or experiment.get('parent_sha256') != parent_sha
            or experiment.get('inference_only') is not True
            or experiment.get('status') != 'complete' or experiment.get('blend')):
        raise ValueError('Initialization requires a completed, unblended BASE probe from the same root')
    for key in ('config', 'tokenizer'):
        if key not in parent or initial.get(key) != parent[key]:
            raise ValueError(f'Initialization {key} contract differs from the frozen root')
    for key in ('step', 'tokens_processed'):
        value = initial.get(key)
        if type(value) is not int or value < parent[key]:
            raise ValueError(f'Invalid cumulative initialization counter: {key}')


def probe_learning_rate(step, total_steps, peak, schedule='constant'):
    """Zero-based update schedule; cosine reaches 10% on the final update."""
    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError('Update index must be within the planned training run')
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError('Peak learning rate must be finite and positive')
    if schedule not in ('constant', 'cosine'):
        raise ValueError('Unknown probe learning-rate schedule')
    if schedule == 'constant' or total_steps == 1:
        return peak
    progress = step / (total_steps - 1)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def answer(op, a, b):
    return {'+': lambda: a+b, '-': lambda: a-b,
            '*': lambda: a*b, '/': lambda: a//b}[op]()


def build_tasks(seed=20260922, *, worked_examples=False):
    from scripts._base_probe_cases import suite
    rng = np.random.default_rng(seed)
    excluded = set()
    if worked_examples:
        from scripts.eval_base_readiness import cases
        for split in ('dev', 'test'):
            excluded.update(pair_key('+', *r['operands']) for r in cases(split) if 'operands' in r)
    for row in suite():
        if row['category'] == 'arithmetic':
            for a, op, b in re.findall(r'(\d+)\s*([+*/-])\s*(\d+)\s*=', row['prompt']):
                excluded.add(pair_key(op, int(a), int(b)))
    tests, pools = [], {}
    words = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'divided by'}
    for op in ['+', '-', '*', '/']:
        pairs = [(a,b) for a in range(1,100) for b in range(1,100)
                 if (op != '-' or a >= b) and (op != '/' or a % b == 0)
                 and (op != '*' or (a <= 12 and b <= 12))]
        # Keep symmetric pairs on the same side of the train/test split.
        by_key = {pair_key(op,a,b):(a,b) for a,b in pairs if pair_key(op,a,b) not in excluded}
        keys = list(by_key)
        rng.shuffle(keys)
        held = set(keys[:24])
        for i,key in enumerate(keys[:24]):
            a,b = by_key[key]
            for fmt in ['equation','words']:
                prompt = f'{a} {op} {b} =' if fmt == 'equation' else f'{a} {words[op]} {b} equals'
                tests.append(dict(id=f'{op}_{i}_{fmt}',category=f'arithmetic_{fmt}',prompt=prompt,
                                  expected_regex=rf'{answer(op,a,b)}(?:\.0+)?(?![\d.])',
                                  max_new_tokens=12, pair=list(key)))
        pools[op] = [by_key[key] for key in keys if key not in held]
    curriculum = []
    for i in range(4096):
        op = ['+', '-', '*', '/'][i % 4]
        pairs = [pools[op][int(rng.integers(len(pools[op])))] for _ in range(6)]
        lines = [f'{a} {op} {b} = {answer(op,a,b)}' for a,b in pairs]
        if i % 3 == 0:
            lines = [f'{a} {words[op]} {b} equals {answer(op,a,b)}.' for a,b in pairs]
        if worked_examples and i % 3 == 1:
            lines = []
            def quantity(n, singular, plural=None):
                return f'{n} {singular if n == 1 else plural or singular + "s"}'
            for a,b in pairs:
                c = answer(op,a,b)
                narrative = {
                    '+': f'A library had {quantity(a,"book")} and received {quantity(b,"book")} more. It now has {quantity(c,"book")}.',
                    '-': f'An orchard contained {quantity(a,"apple")}. After someone picked {quantity(b,"apple")}, {quantity(c,"apple")} remained.',
                    '*': f'A room contains {quantity(a,"shelf","shelves")} with {quantity(b,"cup")} on each shelf. The room holds {quantity(c,"cup")} altogether.',
                    '/': f'A teacher shares {quantity(a,"pencil")} equally among {quantity(b,"student")}. Each student receives {quantity(c,"pencil")}.',
                }[op]
                lines.append(f'{narrative} Calculation: {a} {op} {b} = {c}.')
        curriculum.append({'kind':'arithmetic','text':'\n'.join(lines),
                           'pairs':[list(pair_key(op,a,b)) for a,b in pairs]})
    train_strings = set()
    for i in range(1024):
        text = ''.join(rng.choice(list('abcdefghijklmnpqrstuvwxy'),size=int(rng.integers(1,13))))
        train_strings.add(text)
        a,b,c = (int(x) for x in rng.integers(1,30,size=3))
        lines = [f'>>> len("{text}")\n{len(text)}', f'>>> "{text}".upper()\n"{text.upper()}"',
                 f'>>> sum([{a}, {b}, {c}])\n{a+b+c}', f'>>> [{a}, {b}, {c}][1]\n{b}']
        curriculum.append({'kind':'code','text':'\n'.join(lines), 'string':text})
    for i in range(24):
        # New literal strings, with lengths overlapping the curriculum.
        text = ''.join(rng.choice(list('opqrstuvwxyz'),size=int(rng.integers(2,12))))
        while text in train_strings:
            text += 'z'
        tests.append(dict(id=f'len_{i}',category='code_length',prompt=f'>>> len("{text}")\n',
                          expected_regex=rf'{len(text)}(?!\d)',max_new_tokens=12))
    return curriculum, tests


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    curriculum, tests = build_tasks(worked_examples=args.worked_examples)
    tokenizer = SpakieTokenizer(str(args.assets/'tokenizer/spakie.model'))
    tokens = []
    with (args.output/'curriculum.jsonl').open('w') as stream:
        for row in curriculum:
            stream.write(json.dumps(row)+'\n')
            tokens.extend(tokenizer.encode(row['text'],add_eos=True))
    np.save(args.output/'curriculum.npy', np.asarray(tokens,dtype=np.uint16))
    (args.output/'tasks.json').write_text(json.dumps(tests,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({
        'tokenizer_sha256':digest(args.assets/'tokenizer/spakie.model'),
        'curriculum_sha256':digest(args.output/'curriculum.npy'),
        'tasks_sha256':digest(args.output/'tasks.json'),
        'documents':len(curriculum),'tokens':len(tokens),'seed':20260922, 'worked_examples': args.worked_examples,
        'objective':'All-token next-token prediction, plain completions, no chat roles.'},indent=2)+'\n')
    print(f'Prepared {len(curriculum)} documents / {len(tokens)} tokens / {len(tests)} task probes.')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records = (args.output/'records.jsonl').open('w')
    stop = False
    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True  # finish the in-flight update before snapshotting
    old_handler = signal.signal(signal.SIGINT, request_stop)
    def emit(kind, **values):
        records.write(json.dumps({'kind':kind, **values})+'\n')
        records.flush()
        os.fsync(records.fileno())
    sequence_length = 128 if args.arm == 'sanity' else args.sequence_length
    window = sequence_length + 1
    summary = {'status':'initializing','completed_steps':0,'arm':args.arm,'seed':args.seed,
               'sequence_length':sequence_length,
               'optimizer':args.optimizer, 'replay_kl_weight':args.replay_kl_weight,
               'curriculum_token_presentations':0, 'replay_loss':args.replay_loss,
               'curriculum_sampling':args.curriculum_sampling,
               'curriculum_window':args.curriculum_window}
    model = optimizer = None
    start = time.monotonic()
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from configs.default import CHECKPOINT_CONFIG_SCHEMA_VERSION
        from model.transformer_mlx import SpakieGPTMLX
        from runtime.checkpoint_io import (load_mlx_checkpoint_config,load_mlx_checkpoint_meta,
            load_mlx_model_weights_strict,validate_checkpoint_tokenizer,validate_checkpoint_processed_data)
        from runtime.mlx_backend import load_safetensors,clip_grads,save_safetensors_checkpoint
        from training.optimizers_mlx import configure_mlx_optimizer
        from training.pretrain_mlx import _build_microbatch_step
        from inference.generate_mlx import generate
        from scripts._base_probe_cases import suite,repetition,score_completion

        checkpoint=args.assets/'checkpoints/92m/pretrain_interrupt.safetensors'
        checkpoint_sha=digest(checkpoint)
        expected='792d96759adecfb748575e10e76b03ae8446b784e931a5f88d501f6c3361e35c'
        if checkpoint_sha != expected:
            raise ValueError('Base checkpoint changed; freeze a new experiment before running.')
        config=load_mlx_checkpoint_config(str(checkpoint))
        if sequence_length > config.max_seq_len:
            raise ValueError('Probe sequence length exceeds checkpoint context length')
        metadata=load_mlx_checkpoint_meta(str(checkpoint))
        initial_metadata=metadata
        validate_checkpoint_tokenizer(metadata,str(args.assets/'tokenizer/spakie.model'),source=str(checkpoint))
        validate_checkpoint_processed_data(metadata,str(args.assets/'data/processed'),source=str(checkpoint))
        tokenizer=SpakieTokenizer(str(args.assets/'tokenizer/spakie.model'))
        manifest=json.loads((args.data/'manifest.json').read_text())
        if manifest['tokenizer_sha256'] != digest(args.assets/'tokenizer/spakie.model'):
            raise ValueError('Curriculum tokenizer mismatch')
        for name,key in [('curriculum.npy','curriculum_sha256'),('tasks.json','tasks_sha256')]:
            if digest(args.data/name) != manifest[key]: raise ValueError(f'{name} changed')
        mx.set_cache_limit(512*1024*1024)
        mx.random.seed(args.seed)
        model=SpakieGPTMLX(config)
        flat=load_safetensors(str(checkpoint))
        load_mlx_model_weights_strict(model,flat,path=str(checkpoint));del flat
        if args.initialize_from is not None:
            initial_metadata=load_mlx_checkpoint_meta(str(args.initialize_from))
            validate_probe_initialization(metadata,initial_metadata,checkpoint_sha)
            initial_weights=load_safetensors(str(args.initialize_from))
            if set(initial_weights) != {'model.'+k for k,_ in tree_flatten(model.parameters())}:
                raise ValueError('Initialization must contain only the exact model tensors')
            load_mlx_model_weights_strict(model,initial_weights,path=str(args.initialize_from))
            del initial_weights
            summary['initial_checkpoint']={
                'path':str(args.initialize_from.resolve()),'sha256':digest(args.initialize_from),
                'step':initial_metadata['step'],'tokens_processed':initial_metadata['tokens_processed'],
                'optimizer_state_loaded':False,'anchor_base_sha256':checkpoint_sha}
        if args.blend_from is not None:
            donor_meta=load_mlx_checkpoint_meta(str(args.blend_from))
            if donor_meta.get('experiment',{}).get('parent_sha256') != expected:
                raise ValueError('Blend source must be a probe derived from the same frozen BASE')
            if donor_meta['config'] != metadata['config'] or donor_meta['tokenizer'] != metadata['tokenizer']:
                raise ValueError('Blend source has a different model/tokenizer contract')
            donor=load_safetensors(str(args.blend_from))
            weights={k:v for k,v in tree_flatten(model.parameters())}
            if {'model.'+k for k in weights} != set(donor):
                raise ValueError('Blend source must contain exactly the model tensors')
            blended={}
            for name,base in weights.items():
                other=donor['model.'+name]
                if base.shape!=other.shape:raise ValueError(f'Blend shape mismatch: {name}')
                blended['model.'+name]=((1-args.blend_fraction)*base.astype(mx.float32)+args.blend_fraction*other.astype(mx.float32)).astype(base.dtype)
            load_mlx_model_weights_strict(model,blended,path=str(args.blend_from))
            del donor,weights,blended
        mx.eval(model.parameters())
        initial_weight=mx.array(model.ln_f.weight.astype(mx.float32));mx.eval(initial_weight)
        optimizer=configure_mlx_optimizer(model,config,kind=args.optimizer,learning_rate=args.lr,weight_decay=.1)
        step_fn=_build_microbatch_step(model,1.,compile_step=False,ignore_index=None)
        anchored_step = None
        if args.replay_kl_weight:
            from scripts._replay_anchor import build_anchored_step
            # Preserve the student's stochastic state while constructing a
            # separate, immutable teacher from the exact frozen parent.
            rng_state = [mx.array(value) for value in mx.random.state]
            teacher = SpakieGPTMLX(config)
            load_mlx_model_weights_strict(teacher,load_safetensors(str(checkpoint)),path=str(checkpoint))
            teacher.eval()
            mx.eval(teacher.parameters())
            mx.random.state = rng_state
            anchored_step = build_anchored_step(model,teacher,args.replay_kl_weight,
                                                replace_replay_ce=args.replay_loss=='kl')
        rng=np.random.default_rng(args.seed)
        train=np.load(args.assets/'data/processed/train.npy',mmap_mode='r')
        val=np.load(args.assets/'data/processed/val.npy',mmap_mode='r')
        curriculum=np.load(args.data/'curriculum.npy',mmap_mode='r')
        if len(train) <= window or len(curriculum) <= window:
            raise ValueError('Probe arrays must exceed the requested sequence length plus one')
        document_starts = None
        if args.curriculum_window == 'document' and args.arm == 'curriculum':
            document_starts = verified_document_starts(
                args.data/'curriculum.jsonl', tokenizer, curriculum, window=window, stopped=lambda: stop)
        tasks=json.loads((args.data/'tasks.json').read_text())
        provenance={'base_checkpoint':str(checkpoint),'base_sha256':checkpoint_sha,'data_manifest':manifest,
                    'arm':args.arm,'seed':args.seed,'batch_size':4,'sequence_length':sequence_length,
                    'max_steps':args.steps,'max_training_seconds':args.seconds,'learning_rate':args.lr,
                    'learning_rate_schedule':args.lr_schedule,
                    'planned_final_learning_rate':probe_learning_rate(args.steps-1,args.steps,args.lr,args.lr_schedule),
                    'curriculum_fraction':args.curriculum_fraction,
                    'replay_kl_weight':args.replay_kl_weight,
                    'replay_loss':args.replay_loss,'curriculum_sampling':args.curriculum_sampling,
                    'curriculum_window':args.curriculum_window,
                    'replay_kl_objective':'KL(frozen parent || student), general replay rows only, mean over all batch tokens; temperature1',
                    'optimizer':f'{args.optimizer}, fresh state; BF16 model, FP32 masters','weight_decay':.1,'clip_norm':1.,
                    'loss':('curriculum next-token cross entropy plus replay KL; mean over all batch tokens; no assistant mask'
                            if args.replay_loss=='kl' else
                            'all-token cross entropy plus replay KL; no assistant mask'
                            if args.replay_kl_weight else
                            'all next tokens; repository custom loss and microbatch function; no assistant mask'),
                    'script_sha256':digest(__file__),'generation':{'temperature':0.,'top_k':0,'top_p':1.,'penalty':1.,'seed':42},
                    'config':metadata['config'],'device':mx.device_info()}
        if 'initial_checkpoint' in summary:
            provenance['initial_checkpoint']=summary['initial_checkpoint']
        if document_starts is not None:
            provenance['curriculum_documents'] = {
                'jsonl_sha256':digest(args.data/'curriculum.jsonl'),
                'count':len(document_starts),
                'sampling':'Uniform token draw snapped backward to document start; length-biased, not uniform documents',
                'first_document_complete':True,
                'later_documents_may_be_truncated':True}
        if args.blend_from is not None:
            provenance['blend']={'donor':str(args.blend_from.resolve()),'donor_sha256':digest(args.blend_from),
                                 'donor_fraction':args.blend_fraction,'base_fraction':1-args.blend_fraction,
                                 'arithmetic':'FP32 linear interpolation, rounded to original BF16 storage'}
        (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        (args.output/'probe_base_learning.py').write_text(Path(__file__).read_text())
        helper_paths = [ROOT/'scripts/_base_probe_cases.py']
        if args.replay_kl_weight:
            helper_paths.append(ROOT/'scripts/_replay_anchor.py')
        (args.output/'source_hashes.json').write_text(json.dumps(
            {str(path.relative_to(ROOT)):digest(path) for path in helper_paths},indent=2)+'\n')
        for path in helper_paths:
            (args.output/path.name).write_bytes(path.read_bytes())
        sanity_rows=[('The badge code of Neral is','742.'),('The badge code of Vesk is','315.'),
                     ('The badge code of Pold is','628.'),('The badge code of Jutan is','951.')]
        fixed=[]
        for prompt,completion in sanity_rows:
            ids=tokenizer.encode(prompt+' '+completion,add_eos=True)
            fixed.append((ids*math.ceil(129/len(ids)))[:129])
        fixed=np.array(fixed,dtype=np.int32)
        fixed_tasks=[dict(id=f'sanity_{i}',category='memorization',prompt=p,expected_regex=re.escape(a[:-1])+r'(?!\d)',max_new_tokens=12) for i,(p,a) in enumerate(sanity_rows)]
        (args.output/'sanity_documents.json').write_text(json.dumps(sanity_rows,indent=2)+'\n')

        def evaluate(label):
            model.eval()
            if args.arm=='sanity':
                selected=fixed_tasks
            else:
                indices=np.random.default_rng(20260922).choice((len(val)-1)//2048,size=128,replace=False)[:32]
                losses=[]
                for index in indices:
                    if stop: return
                    chunk=np.array(val[int(index)*2048:int(index)*2048+2049],dtype=np.int32)
                    _,loss,_=model(mx.array(chunk[:-1][None]),targets=mx.array(chunk[1:][None]))
                    losses.append(float(loss.item()))
                emit('validation',phase=label,indices=indices.tolist(),tokens=32*2048,
                     mean_nll=float(np.mean(losses)),window_nll=losses)
                selected=tasks+[r for r in suite() if r['category']!='continuation' or r['id'] in ['continue_story','continue_science','continue_exposition']]
            scores={}
            for row in selected:
                if stop: return
                settings=[(0.,42)]
                if row['category']=='continuation': settings.append((.7,42))
                for temperature,seed in settings:
                    np.random.seed(seed);mx.random.seed(seed)
                    output=generate(model,tokenizer,tokenizer.encode(row['prompt']),max_new_tokens=row['max_new_tokens'],
                        temperature=temperature,top_k=50 if temperature else 0,top_p=.9 if temperature else 1.,repetition_penalty=1.)
                    text=tokenizer.decode(output)
                    passed=score_completion(text,row) if 'expected_regex' in row else None
                    emit('generation',phase=label,**generation_task_fields(row),output=text,output_ids=output,passed=passed,
                         temperature=temperature,seed=seed,repeat_4gram_fraction=repetition(output))
                    if passed is not None:
                        values=scores.setdefault(row['category'],{'passed':0,'total':0})
                        values['passed']+=int(passed);values['total']+=1
            emit('scores',phase=label,scores=scores)
            print(label,json.dumps(scores),flush=True)

        if args.arm in ('baseline','sanity'):evaluate('before')
        if args.arm=='baseline':
            if args.readiness and not stop:
                from scripts.eval_base_readiness import evaluate_model
                summary['readiness'] = evaluate_model(model, args.assets/'tokenizer/spakie.model',
                    args.output/'readiness.json', stopped=lambda: stop)
            summary['status']='interrupted' if stop else 'complete'
            return 130 if stop else 0
        if stop: return 130
        model.train()
        train_start=time.monotonic()
        for i in range(args.steps):
            if stop or time.monotonic()-train_start>=args.seconds:break
            selected_rows = []
            curriculum_offsets = []
            if args.arm=='sanity':
                batch=fixed
                starts=[]
            else:
                starts=rng.integers(0,len(train)-window,size=4).tolist()
                batch=np.stack([train[s:s+window] for s in starts]).astype(np.int32)
                if args.arm=='curriculum':
                    # Independent RNG preserves the replay-row schedule between arms.
                    crng=np.random.default_rng(args.seed*100000+i)
                    if args.curriculum_sampling=='fixed':
                        selected_rows = list(range(4-int(args.curriculum_fraction*4),4))
                    else:
                        selected_rows = (2,3) if args.curriculum_fraction == .5 else np.flatnonzero(crng.random(4) < args.curriculum_fraction)
                    for j in selected_rows:
                        s=int(crng.integers(0,len(curriculum)-window))
                        if document_starts is not None:
                            s=snap_to_document_start(s,document_starts)
                        curriculum_offsets.append(s)
                        batch[j]=curriculum[s:s+window]
            replay_rows = np.ones(4,dtype=bool)
            replay_rows[list(selected_rows)] = False
            if anchored_step is not None:
                (loss,ce_loss,kl_loss),grads=anchored_step(
                    mx.array(batch[:,:-1]),mx.array(batch[:,1:]),mx.array(replay_rows))
                mx.eval(ce_loss,kl_loss)
                components={'cross_entropy':float(ce_loss.item()),'replay_kl':float(kl_loss.item())}
            else:
                loss,grads=step_fn(mx.array(batch[:,:-1]),mx.array(batch[:,1:]))
                components={}
            grads,norm=clip_grads(grads,1.)
            mx.eval(loss,norm,grads)
            loss_value=float(loss.item());norm_value=float(norm.item())
            if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                raise ValueError('Non-finite loss or gradient')
            learning_rate=probe_learning_rate(i,args.steps,args.lr,args.lr_schedule)
            if args.lr_schedule != 'constant':
                optimizer.set_lr(learning_rate)
            optimizer.update(model,grads);optimizer.eval_state()
            summary['completed_steps']=i+1
            summary['curriculum_token_presentations'] += len(selected_rows)*sequence_length
            emit('training',step=i+1,loss=loss_value,gradient_norm=norm_value,learning_rate=learning_rate,
                 curriculum_rows=[int(j) for j in selected_rows],
                 curriculum_offsets=curriculum_offsets,**components,
                 replay_offsets=starts,batch_sha256=hashlib.sha256(batch.tobytes()).hexdigest(),
                 elapsed_seconds=time.monotonic()-train_start)
            if (i+1)%16==0:
                elapsed=time.monotonic()-train_start
                eta=min(args.seconds-elapsed,elapsed/(i+1)*(args.steps-i-1))
                print(f"step {i+1}/{args.steps} loss={loss_value:.4f} elapsed={elapsed:.1f}s ETA={max(0,eta):.0f}s",flush=True)
        summary['training_seconds']=time.monotonic()-train_start
        summary['learning_rate_schedule']=args.lr_schedule
        summary['peak_learning_rate']=args.lr
        summary['last_learning_rate']=(probe_learning_rate(summary['completed_steps']-1,args.steps,args.lr,args.lr_schedule)
                                       if summary['completed_steps'] else None)
        summary['token_presentations']=summary['completed_steps']*4*sequence_length
        summary['layernorm_max_abs_change']=float(mx.max(mx.abs(model.ln_f.weight.astype(mx.float32)-initial_weight)).item())
        summary['peak_memory_gb']=mx.get_peak_memory()/1024**3
        summary['status']='interrupted' if stop else ('complete' if summary['completed_steps']==args.steps else 'time_limit')
        # Persist the trained weights before optional evaluation. A scorer
        # failure must not discard a completed, bounded training experiment.
        # Inference-only experimental snapshot. No optimizer state/resume claim.
        snapshot_meta={k:v for k,v in metadata.items() if k not in ['rng','rng_state','sampler','checkpoint_generation']}
        snapshot_meta['experiment']={**summary,'parent_sha256':checkpoint_sha,'optimizer_restarted':True,'inference_only':True}
        snapshot_meta['stage']='pretrain'
        snapshot_meta['config_schema_version']=CHECKPOINT_CONFIG_SCHEMA_VERSION
        snapshot_meta['step']=initial_metadata['step']+summary['completed_steps']
        snapshot_meta['tokens_processed']=initial_metadata['tokens_processed']+summary['token_presentations']
        snapshot_meta.pop('val_loss',None);snapshot_meta.pop('best_val_loss',None)
        save_safetensors_checkpoint(str(args.output/'base_probe.safetensors'),
            {'model.'+k:v for k,v in tree_flatten(model.parameters())},snapshot_meta)
        if not stop:evaluate('after')
        if args.readiness and not stop:
            from scripts.eval_base_readiness import evaluate_model
            summary['readiness'] = evaluate_model(model, args.assets/'tokenizer/spakie.model',
                args.output/'readiness.json', stopped=lambda: stop)
        if stop:summary['status']='interrupted'
    except KeyboardInterrupt:
        summary['status']='interrupted'
    except Exception as exc:
        summary['status']='failed'
        summary['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if summary['status']=='initializing':summary['status']='interrupted' if stop else 'failed'
        summary['elapsed_seconds']=time.monotonic()-start
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        records.close()
        signal.signal(signal.SIGINT,old_handler)
        print(json.dumps(summary),flush=True)
    return 130 if summary['status']=='interrupted' else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--arm',choices=['baseline','sanity','replay','curriculum'],default='baseline')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--sequence-length',type=int,choices=[256,512,1024,2048],default=256,
                        help='Training context for bounded probes; token budget scales with this length (default 256)')
    parser.add_argument('--seconds',type=int,default=240)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument('--extended-pilot',action='store_true',
                         help='Explicit longer diagnostic: at most 2048 updates / 600 training seconds')
    budgets.add_argument('--token-pilot',action='store_true',
                         help='Explicit larger token budget: at most 4096 updates / 600 training seconds')
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--lr-schedule',choices=['constant','cosine'],default='constant',
                        help='Optional cosine decay to 10%% of peak over the planned updates; default constant')
    parser.add_argument('--optimizer',choices=['muon','adamw'],default='muon',
                        help='Optimizer for the bounded BASE continuation experiment')
    parser.add_argument('--replay-kl-weight',type=float,default=0.,
                        help='Opt-in frozen-parent prediction penalty on general replay rows (default disabled)')
    parser.add_argument('--replay-loss',choices=['ce','kl'],default='ce',
                        help='Use KL instead of hard-label cross entropy on replay rows; requires a positive KL weight')
    parser.add_argument('--curriculum-sampling',choices=['legacy','fixed'],default='legacy',
                        help='Fixed sampling gives exactly 0, 1, or 2 curriculum rows in every four-row batch')
    parser.add_argument('--curriculum-fraction',type=float,default=.5)
    parser.add_argument('--curriculum-window',choices=['random','document'],default='random',
                        help='Opt-in token draw snapped to a verified document start; changes example exposure')
    parser.add_argument('--worked-examples',action='store_true',help='Prepare varied, verified narrative arithmetic completions')
    parser.add_argument('--readiness',action='store_true',help='Also run independent BASE development diagnostics')
    initializers=parser.add_mutually_exclusive_group()
    initializers.add_argument('--blend-from',type=Path,help='Evaluate a partial probe update without more training')
    initializers.add_argument('--initialize-from',type=Path,
                              help='Start a fresh bounded optimizer stage from a completed BASE probe; not resume')
    parser.add_argument('--blend-fraction',type=float,default=.25)
    args=parser.parse_args()
    if args.arm=='sanity' and args.sequence_length!=256:
        parser.error('The fixed sanity probe uses 128-token sequences; do not override sequence length')
    max_steps, max_seconds = ((4096,600) if args.token_pilot else
                              (2048,600) if args.extended_pilot else (256,240))
    if not 1<=args.steps<=max_steps or not 1<=args.seconds<=max_seconds:
        parser.error(f'Hard limits: 1..{max_steps} updates and 1..{max_seconds} training seconds per run')
    if not math.isfinite(args.lr) or not 0 < args.lr <= 1e-4:
        parser.error('Learning rate must be finite and in (0, 1e-4]')
    if not math.isfinite(args.curriculum_fraction) or not 0 <= args.curriculum_fraction <= .5:
        parser.error('Curriculum fraction must be in [0, .5]')
    if not math.isfinite(args.replay_kl_weight) or not 0 <= args.replay_kl_weight <= 20:
        parser.error('Replay KL weight must be finite and in [0, 20]')
    if args.replay_kl_weight and (args.action!='run' or args.arm not in ('replay','curriculum')):
        parser.error('Replay anchoring is only supported by replay/curriculum training probes')
    if args.replay_loss=='kl' and not args.replay_kl_weight:
        parser.error('Replacing replay cross entropy requires a positive replay KL weight')
    if args.curriculum_sampling=='fixed' and args.curriculum_fraction not in (0.,.25,.5):
        parser.error('Fixed sampling supports curriculum fractions 0, 0.25, or 0.5')
    if args.curriculum_window=='document' and (args.action!='run' or args.arm not in ('curriculum','replay')):
        parser.error('Document windows are only supported by curriculum/replay training probes')
    if args.action=='run' and args.data is None:parser.error('--data is required')
    if args.blend_from is not None and (args.action!='run' or args.arm!='baseline' or not 0<args.blend_fraction<1):
        parser.error('Blending is evaluation-only with --arm baseline and a fraction between zero and one')
    if args.initialize_from is not None and (args.action!='run' or args.arm not in ('curriculum','replay')):
        parser.error('Probe initialization is only supported by curriculum/replay training stages')
    # Never allow experiments to create outputs in the original asset tree.
    if args.output.resolve().is_relative_to(args.assets.resolve()):
        parser.error('Output must be outside the original asset directory')
    try:
        return prepare(args) if args.action=='prepare' else run(args)
    except KeyboardInterrupt:
        print('\nStopped; completed files preserved.',file=sys.stderr)
        return 130


if __name__=='__main__':
    raise SystemExit(main())
