"""CPU-only, validation-selected Pythia weight editing. No live service access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,time,copy,platform
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from mergekit.layer_remap import remap_layers
from mergekit.merge_methods import merge_state_dicts
P=Path(__file__).resolve().parent
REV='a39f36b100fe8a5377810d56c3f4789b9c53ac42'
torch.set_num_threads(2);torch.manual_seed(20261118)

def save(name,x):(P/name).write_text(json.dumps(x,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(split):
 docs=[json.loads(x) for x in (P/(split+'.jsonl')).read_text().splitlines()];labs=[int(x) for x in (P/(split+'-labels.lst')).read_text().splitlines()]
 return list(zip(docs,labs))

protocol={'model':'EleutherAI/pythia-70m','revision':REV,'license':'Apache-2.0','threads':2,'dtype':'float32','context':2048,'selection':'first 256 PIQA training examples only','heldout':'all 1838 official validation examples; evaluate once after selection','metric':'zero-shot acc and acc_norm (solution character count), lm-eval v0.3 PIQA format','candidates':['identity']+[f'{kind}_{a}' for kind in ['swap23','shift_neighbors'] for a in [.01,.05,.1]],'weight_training_updates':0,'torch':torch.__version__,'platform':platform.platform(),'data_sha256':{f:digest(P/f) for f in ['train.jsonl','train-labels.lst','valid.jsonl','valid-labels.lst']}}
save('protocol.json',protocol)
tok=AutoTokenizer.from_pretrained(protocol['model'],revision=REV,cache_dir=str(P/'cache'))
m=AutoModelForCausalLM.from_pretrained(protocol['model'],revision=REV,cache_dir=str(P/'cache'),torch_dtype=torch.float32).eval()
original={k:v.detach().clone() for k,v in m.state_dict().items()}
protocol['parameters']=sum(x.numel() for x in m.parameters());protocol['layers']=m.config.num_hidden_layers;save('protocol.json',protocol)
# Pre-tokenize after immutable protocol is saved; no labels used in tokenization.
def encode(data):
 out=[]
 for doc,label in data:
  prompt='Question: '+doc['goal']+'\nAnswer:'
  c=tok.encode(prompt,add_special_tokens=False);choices=[]
  for sol in [doc['sol1'],doc['sol2']]:
   whole=tok.encode(prompt+' '+sol,add_special_tokens=False)
   assert whole[:len(c)]==c,'Tokenizer boundary mismatch'
   assert len(whole)<=2048,'Unexpected truncation'
   choices.append((whole,len(c),len(sol)))
  out.append((choices,label))
 return out
@torch.inference_mode()
def evaluate(data):
 rows=[]
 for choices,label in data:
  scores=[]
  for ids,start,nchars in choices:
   x=torch.tensor([ids],dtype=torch.long);logits=m(x,use_cache=False).logits[0,start-1:-1].float()
   y=x[0,start:]; ll=-torch.nn.functional.cross_entropy(logits,y,reduction='sum').item();scores.append((ll,ll/nchars))
  rows.append({'label':label,'pred':int(scores[1][0]>scores[0][0]),'pred_norm':int(scores[1][1]>scores[0][1]),'scores':scores})
 return {'acc':sum(r['pred']==r['label'] for r in rows)/len(rows),'acc_norm':sum(r['pred_norm']==r['label'] for r in rows)/len(rows),'n':len(rows),'rows':rows}
def candidate(name):
 if name=='identity':return {k:v.clone() for k,v in original.items()}
 kind,alpha=name.rsplit('_',1);n=protocol['layers'];order=list(range(n))
 if kind=='swap23':order[2],order[3]=order[3],order[2]
 else:order=[min(i+1,n-1) for i in range(n)]
 donor=remap_layers(original,order,prefix='gpt_neox.layers')
 return merge_state_dicts([original,donor],'linear',parameters={'weight':[1-float(alpha),float(alpha)]})
# Exact identity control and original storage independence.
identity=remap_layers(original,list(range(protocol['layers'])),prefix='gpt_neox.layers')
assert all(torch.equal(original[k],identity[k]) and original[k].data_ptr()!=identity[k].data_ptr() for k in original)
cal=encode(load('train')[:256]);results={'calibration':{},'heldout':{},'selection_rule':'maximum calibration acc_norm, ties prefer identity then listed order'}
for name in protocol['candidates']:
 t=time.monotonic();m.load_state_dict(candidate(name),strict=True);r=evaluate(cal);results['calibration'][name]=r;save('results_partial.json',results);print('CAL',name,r['acc'],r['acc_norm'],time.monotonic()-t,flush=True)
best=max(protocol['candidates'],key=lambda n:results['calibration'][n]['acc_norm'])
# Select best nonidentity as experimental challenger even if baseline wins: never call it improvement.
challenger=max(protocol['candidates'][1:],key=lambda n:results['calibration'][n]['acc_norm'])
results['selected']=best;results['challenger']=challenger;save('selection.json',{'selected':best,'challenger':challenger,'calibration':{k:{q:v[q] for q in ['acc','acc_norm']} for k,v in results['calibration'].items()}})
test=encode(load('valid'))
for name in ['identity',challenger]:
 t=time.monotonic();m.load_state_dict(candidate(name),strict=True);r=evaluate(test);results['heldout'][name]=r;save('results_partial.json',results);print('HELDOUT',name,r['acc'],r['acc_norm'],time.monotonic()-t,flush=True)
a=np.array([r['pred_norm']==r['label'] for r in results['heldout']['identity']['rows']],dtype=float);b=np.array([r['pred_norm']==r['label'] for r in results['heldout'][challenger]['rows']],dtype=float);d=b-a;rng=np.random.default_rng(20261118);boot=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(10000)])
results['paired']={'delta_acc_norm':float(d.mean()),'bootstrap95':[float(v) for v in np.quantile(boot,[.025,.975])],'wins':int((d>0).sum()),'losses':int((d<0).sum()),'bootstrap_unit':'example; article/source correlations not controlled'}
assert all(torch.equal(original[k],m0) for k,m0 in original.items())
m.load_state_dict(candidate(challenger));m.save_pretrained(P/'challenger');tok.save_pretrained(P/'challenger');save('results.json',results)
print('DONE',results['paired'],flush=True)
