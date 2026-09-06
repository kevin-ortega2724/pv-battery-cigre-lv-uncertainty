from pathlib import Path
import pandas as pd
import pytest

def test_contingency_evidence_separates_disconnection_and_voltage():
    path=Path('results/simulations/contingency_pilot_timeseries.csv')
    if not path.exists(): pytest.skip('Run the contingency pilot first')
    f=pd.read_csv(path,parse_dates=['timestamp'])
    assert f.groupby('case').size().eq(96).all()
    assert f.converged.all()
    e2=f[f['case']=='E2']
    assert e2.event_active.sum()==4
    assert e2.loc[e2.event_active,'deenergized_buses'].eq(17).all()
    assert e2.loc[~e2.event_active,'unserved_kw'].eq(0).all()
    assert e2.unserved_kw.sum()*.25==pytest.approx(40.8469334247)
    e1=f[(f['case']=='E1') & f.event_active]
    assert e1.actual_discharge_kw.eq(0).all()
    assert e1.commanded_discharge_kw.gt(0).all()
    assert e1.unserved_kw.eq(0).all()
    assert f.soc_min.min()>=.1-1e-9 and f.soc_max.max()<=.9+1e-9
