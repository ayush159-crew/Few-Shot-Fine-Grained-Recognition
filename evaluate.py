import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.model import load_model, features, scores

p=argparse.ArgumentParser(); p.add_argument('--data',default='data/CUB_200_2011'); p.add_argument('--episodes',type=int,default=200); p.add_argument('--batch',type=int,default=32); p.add_argument('--out',default='results'); a=p.parse_args()
assert a.episodes>1
start=time.time(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
root=Path(a.data); device='cuda' if torch.cuda.is_available() else 'cpu'
# Fixed custom class-disjoint split; no training or validation tuning in this implementation.
classes=sorted(d for d in (root/'images').iterdir() if d.is_dir()); assert len(classes)==200, 'Expected full CUB dataset'
rng=np.random.default_rng(18); perm=rng.permutation(200); test=[classes[i] for i in perm[150:]]
(out/'split.json').write_text(json.dumps({k:[classes[i].name for i in ids] for k,ids in [('base',perm[:100]),('validation',perm[100:150]),('test',perm[150:])]},indent=2))
torch.manual_seed(42); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
model, transform=load_model(device); bank={}; paths={}
for c in test:
    files=sorted(c.glob('*.jpg')); assert len(files)>=20, c
    paths[c.name]=files; gg=[]; ll=[]
    for i in range(0,len(files),a.batch):
        batch=torch.stack([transform(Image.open(f).convert('RGB')) for f in files[i:i+a.batch]]).to(device)
        g,l=features(model,batch); gg.append(g.cpu()); ll.append(l.cpu())
    bank[c.name]=(torch.cat(gg),torch.cat(ll)); print('Encoded',c.name,flush=True)
methods=['global_euclidean','global_cosine','local_only','global_local']; records=[]; failures=[]
for seed in [42,123,2026]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); print('Seed',seed,flush=True)
    for shot in [1,5]:
        rng=np.random.default_rng(seed)
        for ep in range(a.episodes):
            names=rng.choice(list(bank),5,replace=False); idx=[rng.choice(len(paths[n]),20,replace=False) for n in names]
            sg=torch.stack([bank[n][0][ix[:shot]] for n,ix in zip(names,idx)]); sl=torch.stack([bank[n][1][ix[:shot]] for n,ix in zip(names,idx)])
            qg=torch.cat([bank[n][0][ix[5:]] for n,ix in zip(names,idx)]); ql=torch.cat([bank[n][1][ix[5:]] for n,ix in zip(names,idx)])
            y=torch.arange(5).repeat_interleave(15); gs=scores(sg,qg); ls=scores(sl,ql)
            for method,s in zip(methods,[scores(sg,qg,'euclidean'),gs,ls,(gs+ls)/2]):
                pred=s.argmax(1); records.append([seed,shot,ep,method,float((pred==y).float().mean())])
                if method=='global_local':
                    for j in (pred!=y).nonzero().flatten().tolist():
                        failures.append([seed,shot,ep,str(paths[names[j//15]][idx[j//15][5+j%15]].relative_to(root)),names[y[j]],names[pred[j]]])
def write(name,header,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
write('episodes.csv',['seed','shot','episode','method','accuracy'],records)
write('failures.csv',['seed','shot','episode','image','true','predicted'],failures)
summary=[]
for m in methods:
    for shot in [1,5]:
        v=np.array([r[4] for r in records if r[1]==shot and r[3]==m])*100
        summary.append([m,shot,len(v),v.mean(),1.96*v.std(ddof=1)/np.sqrt(len(v))])
write('main_table.csv',['method','shot','episodes','accuracy_percent','approx_episode_ci95_halfwidth'],summary)
write('ablation.csv',['method','shot','episodes','accuracy_percent','approx_episode_ci95_halfwidth'],[r for r in summary if r[0]!='global_euclidean'])
from collections import Counter
write('failure_pairs.csv',['true','predicted','count'],[(x,y,n) for (x,y),n in Counter((r[4],r[5]) for r in failures).most_common()])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
for shot in [1,5]:
    rows=[r for r in summary if r[1]==shot]; plt.figure(figsize=(8,4)); plt.bar([r[0] for r in rows],[r[3] for r in rows],yerr=[r[4] for r in rows]); plt.ylabel('Accuracy (%)'); plt.ylim(0,100); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(out/f'accuracy_{shot}shot.png'); plt.close()
meta={'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'torch':torch.__version__,'runtime_seconds':time.time()-start,'seeds':[42,123,2026],'episodes_per_seed':a.episodes}; (out/'run_metadata.json').write_text(json.dumps(meta,indent=2)); print(meta)
