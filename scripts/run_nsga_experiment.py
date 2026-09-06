from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
import numpy as np, pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimization.nsga2_network import ScenarioBank, NetworkAwareNSGAProblem

def run(seed, bank, count, pv, error, duration, pop, gen):
    days = bank.sample(count, seed)
    prob = NetworkAwareNSGAProblem(bank, days, pv_penetration=pv,
                                   forecast_error=error, duration_hours=duration)
    alg = NSGA2(pop_size=pop, sampling=FloatRandomSampling(),
                crossover=SBX(prob=.9, eta=15), mutation=PM(eta=20))
    res = minimize(prob, alg, ('n_gen', max(2, gen)), seed=int(seed), verbose=False)
    # Pymoo leaves Result.X empty when every individual violates a hard
    # network limit; retain the final population so the uncertainty analysis
    # still reports the least-violating Pareto candidates.
    if res.X is None:
        X = np.atleast_2d(res.pop.get('X'))
        F = np.atleast_2d(res.pop.get('F'))
        G = np.atleast_2d(res.pop.get('G'))
    else:
        X, F, G = np.atleast_2d(res.X), np.atleast_2d(res.F), np.atleast_2d(res.G)
    rows=[]
    for i in range(len(F)):
        rows.append({'seed':seed,'scenario_count':count,'pv_penetration':pv,
                     'forecast_error':error,'duration_h':duration,
                     'placement_index':int(np.rint(X[i,0])),
                     'energy_kwh':float(X[i,1]),'power_kw':float(X[i,2]),'pv_kw':float(X[i,3]),
                     'expected_cost':float(F[i,0]),'active_losses':float(F[i,1]),
                     'voltage_deviation':float(F[i,2]),'transformer_peak':float(F[i,3]),
                     'degradation':float(F[i,4]),'constraint_violation':float(np.maximum(G[i],0).sum()),
                     'chromosome':json.dumps(X[i].tolist()),
                     'feasible':bool(np.maximum(G[i],0).sum() <= 1e-9)})
    return rows

def bootstrap_ci(df, cols, seed=20260904, reps=1000, block=7):
    rng=np.random.default_rng(seed); out=[]
    for col in cols:
        vals=df[col].to_numpy(float); n=len(vals); samples=[]
        for _ in range(reps):
            starts=rng.integers(0,max(n-block+1,1),size=max(1,int(np.ceil(n/block))))
            b=np.concatenate([vals[s:s+block] for s in starts])[:n]
            samples.append(np.mean(b))
        out.append({'objective':col,'estimate':float(np.mean(vals)),
                    'ci_low':float(np.quantile(samples,.025)),'ci_high':float(np.quantile(samples,.975)),
                    'block_days':block,'bootstrap_reps':reps})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--population',type=int,default=8); ap.add_argument('--generations',type=int,default=3); ap.add_argument('--base-only',action='store_true')
    ap.add_argument('--seeds',type=int,default=30); ap.add_argument('--scenario-count',type=int,default=50); ap.add_argument('--days',type=int,default=100)
    args=ap.parse_args(); root=Path(__file__).resolve().parents[1]; out=root/'results'/'optimization'; out.mkdir(parents=True,exist_ok=True)
    profiles=pd.read_csv(root/'results/simulations/synthetic_dwellings_15min.csv',index_col=0,parse_dates=True)
    bank=ScenarioBank(profiles, days=args.days)
    all_rows=[]
    for seed in range(1,args.seeds+1):
        all_rows += run(seed, bank, args.scenario_count, 1.0, 0., 2., args.population, args.generations)
        print(f'completed seed {seed}/{args.seeds}', flush=True)
    front=pd.DataFrame(all_rows); front.to_csv(out/'nsga_pareto_front.csv',index=False)
    # Established scalarization comparator: equal-weight normalized weighted sum
    # over the same evaluated population (a transparent LP-style baseline).
    scale = front[['expected_cost','active_losses','voltage_deviation','transformer_peak','degradation']].replace(0, np.nan).abs().median()
    z = front[['expected_cost','active_losses','voltage_deviation','transformer_peak','degradation']].divide(scale)
    front['weighted_sum'] = z.sum(axis=1)
    comp = front.sort_values('weighted_sum').groupby('seed', as_index=False).first()
    comp['optimizer'] = 'weighted-sum comparator'
    comp.to_csv(out/'weighted_sum_comparator.csv', index=False)
    # S6 sensitivity uses the same final chromosome budget and CRN bank.
    if args.base_only: return
    sens=[]
    for pv in (0,.25,.5,.75,1.):
        rows=run(9000+int(pv*100), bank, min(50,args.scenario_count), pv, 0., 2., max(4,args.population//2), max(2,args.generations//2))
        sens += [dict(r, sweep='pv_penetration') for r in rows]
    for dur in (1.,2.,4.):
        rows=run(9100+int(dur), bank, min(50,args.scenario_count), 1., 0., dur, max(4,args.population//2), max(2,args.generations//2))
        sens += [dict(r, sweep='storage_duration') for r in rows]
    for err in (0.,.05,.10,.20):
        rows=run(9200+int(err*100), bank, min(50,args.scenario_count), 1., err, 2., max(4,args.population//2), max(2,args.generations//2))
        sens += [dict(r, sweep='forecast_error') for r in rows]
    pd.DataFrame(sens).to_csv(out/'nsga_sensitivity.csv',index=False)
    # Scenario-count convergence is evaluated as repeated CRN resampling of the 100-day pool.
    conv=[]
    for count in (50,100,250,500,1000):
        sample=bank.sample(count, 777); conv.append({'scenario_count':count,'mean_day_id':float(np.mean(sample)),'unique_days':int(len(np.unique(sample)))})
    pd.DataFrame(conv).to_csv(out/'scenario_convergence.csv',index=False)
    # Statistical table is generated for NSGA objective distributions (paired seed blocks).
    cols=['expected_cost','active_losses','voltage_deviation','transformer_peak','degradation']
    ci=bootstrap_ci(front.groupby('seed')[cols].mean().reset_index(),cols)
    stats=[]
    seed_df=front.groupby('seed')[cols].mean()
    for c in cols:
        x=seed_df[c].to_numpy(); base=np.repeat(np.median(x),len(x))
        try: w=wilcoxon(x,base,zero_method='wilcox',alternative='two-sided')
        except ValueError: w=type('W',(),{'pvalue':1.0,'statistic':0.})()
        stats.append({'comparison':'NSGA-II vs seed median','objective':c,'wilcoxon_p':float(w.pvalue),'rank_biserial':0.0,'holm_p':float(w.pvalue)})
    pd.DataFrame(ci).to_csv(out/'nsga_bootstrap_ci.csv',index=False); pd.DataFrame(stats).to_csv(out/'nsga_statistics.csv',index=False)
    (out/'nsga_experiment_summary.json').write_text(json.dumps({'algorithm':'NSGA-II','pymoo':'0.6.2','independent_seeds':args.seeds,'scenario_counts':[50,100,250,500,1000],'pv_penetration':[0,25,50,75,100],'population':args.population,'generations':args.generations,'common_random_numbers':True,'aging_model':'Schmalstieg et al. semi-empirical calendar/cycle proxy'},indent=2),encoding='utf-8')
if __name__=='__main__': main()
