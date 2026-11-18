import json,sys,numpy as np
from pathlib import Path
from scipy.optimize import nnls
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(Path(__file__).parent.parent))
import fit2
from fit2 import Data, load_subs
d=Data();subs=load_subs();nb=d.nbg
print("modes",np.unique(np.concatenate([np.unique(s.prof.mode) for s in subs])),"max cdist",max(int(s.cdist[s.cst>=0].max()) for s in subs))
edges=[0,100,200,300,400,500,650,800,1000,1250,1600,2000]
M=["bus","tram","rail"]
def feats(hmax):
    F=np.zeros((3,len(edges)-1,nb));den=np.zeros(nb)
    for s in subs:
        h=s.prof.h.astype(float);h=np.where(h>0,h,np.inf);hs=h.min(axis=1);mo=s.prof.mode.max(axis=1);ok=s.bg>=0;pw=s.pop.astype(float)
        den+=np.bincount(s.bg[ok],weights=pw[ok],minlength=nb)
        for m in range(3):
            good=(mo==m)&(hs<=hmax)
            dm=np.where((s.cst>=0)&good[np.maximum(s.cst,0)],s.cdist,10**9).min(axis=1)
            for b in range(len(edges)-1):
                sel=(dm>=edges[b])&(dm<edges[b+1])
                F[m,b]+=np.bincount(s.bg[ok],weights=(pw*sel)[ok],minlength=nb)
    return F/np.maximum(den,1e-9)
idx=d.idx;share=d.y/d.n;w=d.w*d.pop[idx];W=np.sqrt(w)
cb=d.cbsa
res={}
for hmax in (1e9,30):
    F=feats(hmax)[:,:,idx];X=np.concatenate([np.ones((1,len(idx))),F.reshape(-1,len(idx))]).T
    coef,_=nnls(X*W[:,None],share*W)
    # cluster bootstrap over metros
    ub=np.unique(cb);rng=np.random.default_rng(0);bs=[]
    for _ in range(30):
        pick=rng.choice(ub,len(ub));ii=np.concatenate([np.flatnonzero(cb==u) for u in pick])
        bs.append(nnls((X*W[:,None])[ii],(share*W)[ii])[0])
    sd=np.std(bs,0);c=coef[1:].reshape(3,-1)*100;s_=sd[1:].reshape(3,-1)*100
    print(f"\n=== headway <= {'any' if hmax>1e8 else int(hmax)} min: effect on transit share (pp) of having the nearest stop of that mode in the band, others fixed (baseline {coef[0]*100:.2f}); +-sd over metros")
    print("band m".ljust(12)+"".join(f"{M[m]:>16}" for m in range(3)))
    for b in range(len(edges)-1):
        print(f"{edges[b]}-{edges[b+1]}".ljust(12)+"".join(f"{c[m,b]:8.2f}+-{s_[m,b]:<6.2f}" for m in range(3)))
    # half distance: first band-centre where effect < half of the effect in bands 0-200 (mean)
    for m in range(3):
        ref=np.mean(c[m,:2]);cen=[(edges[b]+edges[b+1])/2 for b in range(len(edges)-1)]
        half=next((cen[b] for b in range(2,len(cen)) if c[m,b]<ref/2),None)
        print(f"  {M[m]}: effect in 0-200 m {ref:.2f} pp, half-effect at about {half} m")
    res[str(hmax)]={"coef":coef.tolist(),"sd":sd.tolist()}
Path("data/dist_by_mode.json").write_text(json.dumps({"edges":edges,"res":res}))
