"""Check trained weights in two backends and compare cached logits."""
import sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import mlx.core as mx
import torch
from runtime.checkpoint_io import load_mlx_checkpoint_config,load_mlx_model_weights_strict
from runtime.mlx_backend import load_safetensors
from model.transformer_mlx import SpakieGPTMLX
from model.transformer import SpakieGPT
from tokenizer.train_tokenizer import SpakieTokenizer

def main():
 path='checkpoints/92m/pretrain_interrupt.safetensors'
 cfg=load_mlx_checkpoint_config(path);flat=load_safetensors(path)
 m=SpakieGPTMLX(cfg);load_mlx_model_weights_strict(m,flat,path=path);m.set_dtype(mx.float32);m.eval()
 t=SpakieGPT(cfg);state={k[6:]:torch.from_numpy(np.array(v.astype(mx.float32))) for k,v in flat.items() if k.startswith('model.')}
 state['lm_head.weight']=state['tok_emb.weight']
 t.load_state_dict(state,strict=True);t.eval();torch.set_num_threads(4)
 tok=SpakieTokenizer(cfg.tokenizer_prefix+'.model');results=[]
 for text in ['The capital of France is','The chemical formula for water is','17 + 26 =','The capital of France is Paris. It is known for']:
  ids=tok.encode(text); full,_,_=m(mx.array([ids]));_,_,cache=m(mx.array([ids[:-1]]),return_cache=True)
  cached,_,_=m(mx.array([ids[-1:]]),cache=cache,cache_offset=len(ids)-1,return_cache=True)
  with torch.no_grad(): tl=t(torch.tensor([ids]))[0][0,-1].numpy()
  a=np.array(full[0,-1]);b=np.array(cached[0,-1]); order=np.argsort(a)[-5:][::-1]
  results.append(dict(prompt=text,logit_min=float(a.min()),logit_max=float(a.max()),cache_max_diff=float(abs(a-b).max()),torch_max_diff=float(abs(a-tl).max()),argmax=[int(x.argmax()) for x in [a,b,tl]],top5=[{'token':tok.decode([int(i)]),'logit':float(a[i])} for i in order]))
  # Restore full checkpoint weights for each dtype, avoiding irreversible casts.
  m.set_dtype(mx.bfloat16)
  bf,_,_=m(mx.array([ids]));bf=np.array(bf[0,-1].astype(mx.float32))
  results[-1]['bf16_unique_logits']=len(np.unique(bf))
  results[-1]['fp32_unique_logits']=len(np.unique(a))
  results[-1]['bf16_top_logit_ties']=int((bf==bf.max()).sum())
  results[-1]['bf16_argmax']=int(bf.argmax())
  results[-1]['bf16_top20_unique_values']=len(np.unique(np.sort(bf)[-20:]))
  results[-1]['penalty_1_2_top_token_logit_drop']=float(a.max()*0.2)
  load_mlx_model_weights_strict(m,flat,path=path);m.set_dtype(mx.float32)
 (Path(os.environ.get('SPAKIE_DIAGNOSTIC_OUT', str(Path(__file__).parent))) / 'numerics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':
 try:main()
 except KeyboardInterrupt:raise SystemExit(130)
