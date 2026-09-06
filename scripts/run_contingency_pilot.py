"""Controlled 15-min outage replay, not a transient or short-circuit study."""
from pathlib import Path
import sys
import json
import time
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.powerflow.cigre import build_cigre_lv
from src.powerflow.timeseries import normalized_load_multipliers
from src.powerflow.der_timeseries import synthetic_pv_profile
from build_methodology_assets import tex_table

def main():
    started=time.perf_counter()
    sim=Path('results/simulations')
    profiles=pd.read_csv(sim/'synthetic_dwellings_15min.csv',index_col=0,parse_dates=True)
    base=build_cigre_lv(); multipliers=normalized_load_multipliers(profiles,len(base.load))
    # Explicit selection: strongest S2 discharge day; same realizations for all cases.
    day_index=33; ix=np.arange(day_index*96,(day_index+1)*96)
    availability=synthetic_pv_profile(multipliers.index).iloc[ix]
    rows=[]
    for case in ['E0','E1','E2']:
        net=build_cigre_lv(); p0=net.load.p_mw.to_numpy().copy();q0=net.load.q_mvar.to_numpy().copy()
        rid=net.load.index[net.load.name.str.startswith('Load R')].to_numpy()
        pvkw=.5*p0[rid]*1000; cap=2*pvkw; energy=.1*cap
        ids=[pp.create_sgen(net,int(net.load.at[i,'bus']),0,q_mvar=0,name=f'PV_BESS_{i}') for i in rid]
        for step,(stamp,scale) in enumerate(multipliers.iloc[ix].iterrows()):
            net.load.loc[:,'p_mw']=p0*scale.to_numpy(); net.load.loc[:,'q_mvar']=q0*scale.to_numpy()
            active=18<=stamp.hour<19
            line_out=case=='E2' and active
            net.line.at[0,'in_service']=not line_out
            unsupplied=top.unsupplied_buses(net)
            loadkw=net.load.p_mw.to_numpy()[rid]*1000
            solar=pvkw*availability.iloc[step]
            command_c=np.minimum.reduce([np.maximum(solar-loadkw,0),pvkw,np.maximum(.9*cap-energy,0)/(.95*.25)])
            command_d=np.zeros(len(rid))
            if 18<=stamp.hour<22:
                hours=22-stamp.hour-stamp.minute/60
                command_d=np.minimum.reduce([loadkw,pvkw,np.maximum(energy-.1*cap,0)*.95/hours])
            enabled=np.array([int(net.load.at[i,'bus']) not in unsupplied for i in rid])
            battery_enabled=enabled & ~(np.repeat(case=='E1' and active,len(rid)))
            actual_c=command_c*battery_enabled; actual_d=command_d*battery_enabled
            energy+=.95*actual_c*.25-actual_d*.25/.95
            net.sgen.loc[ids,'in_service']=enabled
            net.sgen.loc[ids,'p_mw']=(solar+actual_d-actual_c)/1000
            pp.runpp(net,numba=False,init='flat',tolerance_mva=1e-8,calculate_voltage_angles=True)
            shed=net.load.loc[net.load.bus.isin(unsupplied),'p_mw'].sum()*1000
            rows.append(dict(case=case,timestamp=str(stamp),event_active=active and case!='E0',
                commanded_discharge_kw=float(command_d.sum()),actual_discharge_kw=float(actual_d.sum()),
                charge_kw=float(actual_c.sum()),soc_min=float((energy/cap).min()),soc_max=float((energy/cap).max()),
                minimum_energized_voltage_pu=float(net.res_bus.vm_pu.min()),
                deenergized_buses=len(unsupplied),unserved_kw=float(shed),
                grid_import_kw=float(net.res_ext_grid.p_mw.sum()*1000),converged=bool(net.converged)))
        print(case,'completed',flush=True)
    f=pd.DataFrame(rows);f.to_csv(sim/'contingency_pilot_timeseries.csv',index=False)
    summaries=[]
    for case,g in f.groupby('case'):
        summaries.append({'Case':case,'AC solves':len(g),'Unserved (kWh)':g.unserved_kw.sum()*.25,
                          'Min live V':g.minimum_energized_voltage_pu.min(),
                          'Max off buses':int(g.deenergized_buses.max())})
    s=pd.DataFrame(summaries);s.to_csv('results/tables/contingency_summary.csv',index=False)
    tex_table(s,'contingency_results','One-day controlled outage replay: E0 reference, E1 battery unavailable, E2 first residential line open (18:00--19:00).')
    fig,axs=plt.subplots(3,1,figsize=(8,6),sharex=True,layout='constrained')
    for case,color in zip(['E0','E1','E2'],['#0072B2','#E69F00','#D55E00']):
        g=f[f['case']==case];h=np.arange(96)/4
        axs[0].step(h,g.actual_discharge_kw,where='post',label=case,color=color)
        axs[1].plot(h,g.grid_import_kw,color=color)
        axs[2].step(h,g.unserved_kw,where='post',color=color)
    for ax in axs: ax.axvspan(18,19,color='.6',alpha=.18);ax.grid(alpha=.2)
    command=f[f['case']=='E1'].commanded_discharge_kw
    axs[0].step(np.arange(96)/4,command,where='post',ls='--',lw=1,color='black',label='E1 command')
    axs[0].legend(ncol=3,frameon=False);axs[0].set_ylabel('BESS discharge (kW)')
    axs[1].set_ylabel('Grid import (kW)');axs[2].set_ylabel('Unserved load (kW)')
    axs[2].set(xlim=(16,23),xlabel='Synthetic hour of day')
    fig.savefig('paper/figures/contingency_response.pdf',bbox_inches='tight')
    fig.savefig('paper/figures/contingency_response.png',dpi=180,bbox_inches='tight')
    assert f.converged.all() and f.soc_min.min()>=.1-1e-9 and f.soc_max.max()<=.9+1e-9
    summary={'synthetic_day_index':day_index,'cases':summaries,'solves':len(f),'wall_seconds':time.perf_counter()-started,
             'outage_window':'18:00 <= t < 19:00','line_index':0,'line_name':str(base.line.at[0,'name']),
             'scope':'quasi-static outages; no fault-current, protection, islanding or transient simulation',
             'battery_initial_soc':.1,'no_islanding_supply':True}
    (sim/'contingency_pilot_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
