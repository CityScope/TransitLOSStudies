import sys,json,time,os
import numpy as np
sys.path.insert(0,'fit'); sys.path.insert(0,'.')
import fit2
from transitlos.flex import xp_scoring as X, backend as B
data=fit2.Data(wfile="bg_weights_coarse.npy"); subs=fit2.load_subs(sample_dir=fit2.HERE/"data"/"sample_coarse"); nbg=data.nbg
print("coarse cells",sum(len(s.bg) for s in subs))
sets=[]
for agg in ("nested","power"):
    m=fit2.Model("param",agg); sets.append((agg+" default",m,m.x0()))
    d=json.load(open(f"fit/data/w_param_{agg}_s1.json")); r=min(d["runs"],key=lambda r:r["loss"]); sets.append((agg+" fitted",m,np.array(r["x"])))
    rng=np.random.default_rng(5)
    for i in range(3): sets.append((f"{agg} random{i}",m,m.x0()+rng.normal(0,0.8,len(m.x0()))))
ev={name:X.XPData(subs,nbg,xp) for name,xp in (("numpy",np),("auto",B.xp()))}
print("auto backend:",ev["auto"].xp.__name__)
for name,m,x in sets:
    p=m.params(x); pj=json.loads(p.to_json()); 
    t=time.time(); ref=fit2.predictor(subs,p,nbg); tc=time.time()-t
    out=[]
    for k,e in ev.items():
        e.numerator(pj)   # warm up (kernel compilation)
        t=time.time(); num=e.numerator(pj); out.append((k,time.time()-t,np.abs(num-ref).max()/max(ref.max(),1)))
    print(f"{name:16s} C core {tc*1e3:7.1f} ms | "+" | ".join(f"{k} {tt*1e3:7.1f} ms rel.err {er:.1e}" for k,tt,er in out))
