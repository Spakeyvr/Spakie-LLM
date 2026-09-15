"""Check production mode defaults across three fixed random seeds."""
from run import *
from inference.chat_mlx import _build_prompt_ids

def main():
 with (OUT/'sampling.jsonl').open('w') as out:
  for name in ['pretrain_interrupt','sft_interrupt']:
   path='checkpoints/92m/'+name+'.safetensors';cfg=load_mlx_checkpoint_config(path)
   model=SpakieGPTMLX(cfg);flat=load_safetensors(path);load_mlx_model_weights_strict(model,flat,path=path);del flat
   model.set_dtype(mx.bfloat16);model.eval();tok=SpakieTokenizer(cfg.tokenizer_prefix+'.model')
   chatmode=name.startswith('sft')
   for ident,kind,raw,chat,choices in CASES:
    if ident not in ['france','japan','water','addition','copy','prose']:continue
    prompt=chat if chatmode else raw
    ids=_build_prompt_ids(tok,[{'role':'user','content':prompt}],'') if chatmode else tok.encode(prompt)
    for seed in [14,29,71]:
     np.random.seed(seed);mx.random.seed(seed)
     result=generate(model,tok,ids,max_new_tokens=64,temperature=.1 if chatmode else .8,top_k=1 if chatmode else 50,top_p=1 if chatmode else .9,repetition_penalty=1.2 if chatmode else 1,stop_on_special_tokens=chatmode,ban_special_tokens=chatmode)
     row=dict(checkpoint=name,seed=seed,id=ident,prompt=prompt,answer=tok.decode(result))
     out.write(json.dumps(row)+'\n');out.flush();print(json.dumps(row),flush=True)

if __name__=='__main__':
 try:main()
 except KeyboardInterrupt:raise SystemExit(130)
