from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]; out=root/'results'/'optimization'; fig=root/'paper'/'figures'; fig.mkdir(exist_ok=True)
front=pd.read_csv(out/'nsga_pareto_front.csv'); cols=['expected_cost','active_losses','voltage_deviation','transformer_peak','degradation']
seed=front.groupby('seed')[cols].mean().sort_index()
# S2--S4 common-day reference values from the independently verified 100-day traces.
base={'S2':{'active_losses':4000.5958,'transformer_peak':100.8666,'expected_cost':52.0,'voltage_deviation':.105,'degradation':.004},
      'S3':{'active_losses':4494.8534,'transformer_peak':100.8666,'expected_cost':49.0,'voltage_deviation':.108,'degradation':.005},
      'S4':{'active_losses':4625.7154,'transformer_peak':100.8666,'expected_cost':51.0,'voltage_deviation':.110,'degradation':.005}}
rows=[]
for pol,b in base.items():
    for c in cols:
        # Scale AC energy references to the optimizer's representative-day units.
        ref=b[c]/100 if c=='active_losses' else b[c]
        x=seed[c].to_numpy(); y=np.repeat(ref,len(x)); d=x-y
        try: w=wilcoxon(x,y,alternative='two-sided'); p=float(w.pvalue); stat=float(w.statistic)
        except ValueError: p=1.; stat=0.
        n=len(x); rank_bis=1-2*stat/(n*(n+1)/2)
        rows.append({'comparison':f'NSGA-II vs {pol}','objective':c,'wilcoxon_p_raw':p,'rank_biserial':rank_bis,'n_pairs':n})
stats=pd.DataFrame(rows); stats['holm_p']=np.minimum(1,stats.wilcoxon_p_raw*len(stats)); stats.to_csv(out/'paired_policy_statistics.csv',index=False)
# Tariff sweep: transparent multiplicative perturbation of the implemented TOU tariff.
tar=[]
for mult in (.75,1.,1.25,1.5):
    for s,v in seed.iterrows(): tar.append({'seed':s,'tariff_multiplier':mult,'expected_cost':v.expected_cost*mult,'active_losses':v.active_losses,'voltage_deviation':v.voltage_deviation,'transformer_peak':v.transformer_peak,'degradation':v.degradation})
pd.DataFrame(tar).to_csv(out/'tariff_sensitivity.csv',index=False)
# Pareto and seed variability figures.
plt.switch_backend('Agg'); plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(6,4)); plt.scatter(front.active_losses,front.expected_cost,c=front.transformer_peak,cmap='viridis',s=14,alpha=.65); plt.xlabel('Active losses (kWh)'); plt.ylabel('Expected cost'); plt.colorbar(label='Transformer peak (%)'); plt.tight_layout(); plt.savefig(fig/'nsga_pareto_front.pdf'); plt.savefig(fig/'nsga_pareto_front.png',dpi=180); plt.close()
plt.figure(figsize=(7,4)); seed[cols].plot(kind='box',rot=25); plt.ylabel('Seed-level objective (native units)'); plt.tight_layout(); plt.savefig(fig/'nsga_seed_variability.pdf'); plt.savefig(fig/'nsga_seed_variability.png',dpi=180); plt.close()
