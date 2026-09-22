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


def answer(op, a, b):
    return {'+': lambda: a+b, '-': lambda: a-b,
            '*': lambda: a*b, '/': lambda: a//b}[op]()


def build_tasks(seed=20260922):
    from scripts._base_probe_cases import suite
    rng = np.random.default_rng(seed)
    excluded = set()
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
    curriculum, tests = build_tasks()
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
        'documents':len(curriculum),'tokens':len(tokens),'seed':20260922,
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
    summary = {'status':'initializing','completed_steps':0,'arm':args.arm,'seed':args.seed}
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
        from scripts._base_probe_cases import suite,repetition,score

        checkpoint=args.assets/'checkpoints/92m/pretrain_interrupt.safetensors'
        checkpoint_sha=digest(checkpoint)
        expected='792d96759adecfb748575e10e76b03ae8446b784e931a5f88d501f6c3361e35c'
        if checkpoint_sha != expected:
            raise ValueError('Base checkpoint changed; freeze a new experiment before running.')
        config=load_mlx_checkpoint_config(str(checkpoint))
        metadata=load_mlx_checkpoint_meta(str(checkpoint))
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
        optimizer=configure_mlx_optimizer(model,config,kind='muon',learning_rate=1e-4,weight_decay=.1)
        step_fn=_build_microbatch_step(model,1.,compile_step=False,ignore_index=None)
        rng=np.random.default_rng(args.seed)
        train=np.load(args.assets/'data/processed/train.npy',mmap_mode='r')
        val=np.load(args.assets/'data/processed/val.npy',mmap_mode='r')
        curriculum=np.load(args.data/'curriculum.npy',mmap_mode='r')
        tasks=json.loads((args.data/'tasks.json').read_text())
        provenance={'base_checkpoint':str(checkpoint),'base_sha256':checkpoint_sha,'data_manifest':manifest,
                    'arm':args.arm,'seed':args.seed,'batch_size':4,'sequence_length':128 if args.arm=='sanity' else 256,
                    'max_steps':args.steps,'max_training_seconds':args.seconds,'learning_rate':1e-4,
                    'optimizer':'muon, fresh state; BF16 model, FP32 masters','weight_decay':.1,'clip_norm':1.,
                    'loss':'all next tokens; repository custom loss and microbatch function; no assistant mask',
                    'script_sha256':digest(__file__),'generation':{'temperature':0.,'top_k':0,'top_p':1.,'penalty':1.,'seed':42},
                    'config':metadata['config'],'device':mx.device_info()}
        if args.blend_from is not None:
            provenance['blend']={'donor':str(args.blend_from.resolve()),'donor_sha256':digest(args.blend_from),
                                 'donor_fraction':args.blend_fraction,'base_fraction':1-args.blend_fraction,
                                 'arithmetic':'FP32 linear interpolation, rounded to original BF16 storage'}
        (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        (args.output/'probe_base_learning.py').write_text(Path(__file__).read_text())
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
                    passed=score(text,row['expected_regex']) if 'expected_regex' in row else None
                    emit('generation',phase=label,**row,output=text,output_ids=output,passed=passed,
                         temperature=temperature,seed=seed,repeat_4gram_fraction=repetition(output))
                    if passed is not None:
                        values=scores.setdefault(row['category'],{'passed':0,'total':0})
                        values['passed']+=int(passed);values['total']+=1
            emit('scores',phase=label,scores=scores)
            print(label,json.dumps(scores),flush=True)

        if args.arm in ('baseline','sanity'):evaluate('before')
        if args.arm=='baseline':
            summary['status']='interrupted' if stop else 'complete'
            return 130 if stop else 0
        if stop: return
        model.train()
        train_start=time.monotonic()
        for i in range(args.steps):
            if stop or time.monotonic()-train_start>=args.seconds:break
            if args.arm=='sanity':
                batch=fixed
                starts=[]
            else:
                starts=rng.integers(0,len(train)-257,size=4).tolist()
                batch=np.stack([train[s:s+257] for s in starts]).astype(np.int32)
                if args.arm=='curriculum':
                    # Independent RNG preserves the replay-row schedule between arms.
                    crng=np.random.default_rng(args.seed*100000+i)
                    for j in (2,3):
                        s=int(crng.integers(0,len(curriculum)-257))
                        batch[j]=curriculum[s:s+257]
            loss,grads=step_fn(mx.array(batch[:,:-1]),mx.array(batch[:,1:]))
            grads,norm=clip_grads(grads,1.)
            mx.eval(loss,norm,grads)
            loss_value=float(loss.item());norm_value=float(norm.item())
            if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                raise ValueError('Non-finite loss or gradient')
            optimizer.update(model,grads);optimizer.eval_state()
            summary['completed_steps']=i+1
            emit('training',step=i+1,loss=loss_value,gradient_norm=norm_value,
                 replay_offsets=starts,batch_sha256=hashlib.sha256(batch.tobytes()).hexdigest(),
                 elapsed_seconds=time.monotonic()-train_start)
            if (i+1)%16==0:
                elapsed=time.monotonic()-train_start
                eta=min(args.seconds-elapsed,elapsed/(i+1)*(args.steps-i-1))
                print(f"step {i+1}/{args.steps} loss={loss_value:.4f} elapsed={elapsed:.1f}s ETA={max(0,eta):.0f}s",flush=True)
        summary['training_seconds']=time.monotonic()-train_start
        summary['token_presentations']=summary['completed_steps']*4*(128 if args.arm=='sanity' else 256)
        summary['layernorm_max_abs_change']=float(mx.max(mx.abs(model.ln_f.weight.astype(mx.float32)-initial_weight)).item())
        summary['peak_memory_gb']=mx.get_peak_memory()/1024**3
        summary['status']='interrupted' if stop else ('complete' if summary['completed_steps']==args.steps else 'time_limit')
        if not stop:evaluate('after')
        if stop:summary['status']='interrupted'
        # Inference-only experimental snapshot. No optimizer state/resume claim.
        snapshot_meta={k:v for k,v in metadata.items() if k not in ['rng','rng_state','sampler','checkpoint_generation']}
        snapshot_meta['experiment']={**summary,'parent_sha256':checkpoint_sha,'optimizer_restarted':True,'inference_only':True}
        snapshot_meta['config_schema_version']=CHECKPOINT_CONFIG_SCHEMA_VERSION
        snapshot_meta['step']=metadata['step']+summary['completed_steps']
        snapshot_meta['tokens_processed']=metadata['tokens_processed']+summary['token_presentations']
        snapshot_meta.pop('val_loss',None);snapshot_meta.pop('best_val_loss',None)
        save_safetensors_checkpoint(str(args.output/'base_probe.safetensors'),
            {'model.'+k:v for k,v in tree_flatten(model.parameters())},snapshot_meta)
    except KeyboardInterrupt:
        summary['status']='interrupted'
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
    parser.add_argument('--seconds',type=int,default=240)
    parser.add_argument('--blend-from',type=Path,help='Evaluate a partial probe update without more training')
    parser.add_argument('--blend-fraction',type=float,default=.25)
    args=parser.parse_args()
    if not 1<=args.steps<=256 or not 1<=args.seconds<=240:
        parser.error('Hard limits: 1..256 updates and 1..240 training seconds per run')
    if args.action=='run' and args.data is None:parser.error('--data is required')
    if args.blend_from is not None and (args.action!='run' or args.arm!='baseline' or not 0<args.blend_fraction<1):
        parser.error('Blending is evaluation-only with --arm baseline and a fraction between zero and one')
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
