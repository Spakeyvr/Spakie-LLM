"""Reproducible, local diagnostic; writes each result immediately; Ctrl+C exits cleanly."""
import json
import os
import sys
import hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
from runtime.checkpoint_io import load_mlx_checkpoint_config, load_mlx_checkpoint_meta, load_mlx_model_weights_strict, validate_checkpoint_tokenizer
from runtime.mlx_backend import load_safetensors
from model.transformer_mlx import SpakieGPTMLX
from tokenizer.train_tokenizer import SpakieTokenizer
from inference.generate_mlx import generate
from inference.chat_mlx import _build_prompt_ids

OUT = Path(os.environ.get("SPAKIE_DIAGNOSTIC_OUT", str(Path(__file__).parent)))
OUT.mkdir(parents=True, exist_ok=True)
CASES = [
 ('france','fact','The capital of France is','What is the capital of France?',['Paris','Aubusson','London']),
 ('japan','fact','The capital of Japan is','What is the capital of Japan?',['Tokyo','Kyoto','Paris']),
 ('water','fact','The chemical formula for water is','What is the chemical formula for water?',['H2O','CO2','NaCl']),
 ('planet','fact','The largest planet in our solar system is','What is the largest planet in our solar system?',['Jupiter','Earth','Mars']),
 ('heart','fact','The organ that pumps blood around the human body is the','Which organ pumps blood around the human body?',['heart','liver','brain']),
 ('addition','math','17 + 26 =','What is 17 + 26?',['43','33','42']),
 ('multiply','math','7 * 8 =','What is 7 times 8?',['56','54','48']),
 ('discount','math','A bag costs 60 euros. A 25% discount reduces its price to','A bag costs 60 euros and is discounted by 25%. What is the sale price?',['45','15','35']),
 ('context','grounding','Mira put a green key in a wooden box. The key is colored','Mira put a green key in a wooden box. What color is the key?',['green','wooden','blue']),
 ('counterfact','grounding','In this fictional world, the capital of France is Luma. In this world, the capital of France is','In this fictional world, the capital of France is Luma. What is its capital in this world?',['Luma','Paris','Aubusson']),
 ('order','reasoning','Ava is taller than Ben. Ben is taller than Cleo. The tallest person is','Ava is taller than Ben. Ben is taller than Cleo. Who is tallest?',['Ava','Ben','Cleo']),
 ('python','code','>>> len([4, 8, 12])\n','What does Python print for print(len([4, 8, 12]))?',['3','12','24']),
 ('copy','instruction','The word lantern written in uppercase is','Write lantern in uppercase. Output only the word.',['LANTERN','lantern','LIGHT']),
 ('json','instruction','A JSON object with name set to Nia and age set to 8 is','Return only a JSON object with name Nia and age 8.',['{"name":"Nia","age":8}','Nia is 8','{"name":"Nia","age":9}']),
 ('unknown','calibration','The number of coins in an unopened box cannot be determined without','How many coins are in the closed box on my desk?',['more information','seven coins','ten coins']),
 ('prose','fluency','Rain forms when','Explain how rain forms in two short sentences.',['water vapor','rocks','sunlight']),
]

def main():
 with (OUT/'results.jsonl').open('w') as out:
  for name in ['pretrain_interrupt','sft_interrupt']:
   path = 'checkpoints/92m/'+name+'.safetensors'
   cfg=load_mlx_checkpoint_config(path); meta=load_mlx_checkpoint_meta(path)
   tok=SpakieTokenizer(cfg.tokenizer_prefix+'.model')
   validate_checkpoint_tokenizer(meta,cfg.tokenizer_prefix+'.model',source=path)
   model=SpakieGPTMLX(cfg); flat=load_safetensors(path)
   load_mlx_model_weights_strict(model,flat,path=path); del flat
   model.set_dtype(mx.bfloat16); model.eval(); mx.eval(model.parameters())
   snapshot={k:v for k,v in meta.items() if k not in ['rng_state','sampler_epoch_rng_state']}
   snapshot['file_sha256']=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()
   (OUT/(name+'_metadata.json')).write_text(json.dumps(snapshot,indent=2,default=str))
   for ident,kind,raw,chat,choices in CASES:
    for mode in ['continue','chat']:
     prompt=raw if mode=='continue' else chat
     ids=tok.encode(prompt) if mode=='continue' else _build_prompt_ids(tok,[{'role':'user','content':prompt}],'')
     for penalty in [1.0,1.2]:
      np.random.seed(14); mx.random.seed(14)
      generated=generate(model,tok,ids,max_new_tokens=64,temperature=0,top_k=1,top_p=1,repetition_penalty=penalty,stop_on_special_tokens=True,ban_special_tokens=(mode=='chat'))
      row=dict(checkpoint=name,id=ident,category=kind,mode=mode,penalty=penalty,prompt=prompt,answer=tok.decode(generated),tokens=len(generated))
      out.write(json.dumps(row)+'\n'); out.flush()
      print(json.dumps(row),flush=True)
   # Cache versus full-prefix check, independent of sampling and repetition penalties.
   ids=tok.encode('The capital of France is Paris. It is known for')
   full, _, _ = model(mx.array([ids]))
   _,_,cache=model(mx.array([ids[:-1]]),return_cache=True)
   last,_,_=model(mx.array([ids[-1:]]),cache=cache,cache_offset=len(ids)-1,return_cache=True)
   a=np.asarray(full[0,-1].astype(mx.float32)); b=np.asarray(last[0,-1].astype(mx.float32))
   parity=dict(checkpoint=name,max_logit_difference=float(np.max(np.abs(a-b))),same_argmax=bool(a.argmax()==b.argmax()),finite=bool(np.isfinite(a).all() and np.isfinite(b).all()))
   (OUT/(name+'_cache_check.json')).write_text(json.dumps(parity,indent=2))
   print(parity,flush=True)
   # Fresh, uniformly sampled validation windows; explicit float32 CE.
   val=np.load('data/processed/val.npy',mmap_mode='r'); rng=np.random.default_rng(1409)
   losses=[]
   for start in rng.integers(0,len(val)-513,size=16):
    seq=mx.array(np.asarray(val[start:start+513],dtype=np.int32)[None,:])
    logits,_,_=model(seq[:,:-1]); logits=logits.astype(mx.float32)
    loss=mx.mean(mx.logsumexp(logits,axis=-1)-mx.take_along_axis(logits,seq[:,1:,None],axis=-1).squeeze(-1))
    losses.append(float(loss.item()))
   (OUT/(name+'_validation.json')).write_text(json.dumps({'window_tokens':512,'windows':16,'seed':1409,'mean_loss':float(np.mean(losses)),'window_losses':losses},indent=2))
   # A small precision and cache-free generation control.
   for dtype in [mx.bfloat16,mx.float32]:
    model.set_dtype(dtype)
    for prompt in ['The capital of France is','The chemical formula for water is','17 + 26 =']:
     prefix=tok.encode(prompt); generated=[]
     for _ in range(16):
      logits,_,_=model(mx.array([prefix+generated])); token=int(mx.argmax(logits[0,-1]).item())
      if token==tok.eos_id: break
      generated.append(token)
     control={'checkpoint':name,'dtype':str(dtype),'no_cache':True,'prompt':prompt,'answer':tok.decode(generated)}
     with (OUT/'controls.jsonl').open('a') as f:f.write(json.dumps(control)+'\n')
     print(control,flush=True)
   del model
   mx.clear_cache()

if __name__=='__main__':
 try: main()
 except KeyboardInterrupt:
  print('Interrupted; completed results retained.',flush=True)
  raise SystemExit(130)
