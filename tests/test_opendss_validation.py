from src.powerflow.cigre import build_cigre_lv
from src.powerflow.opendss_validation import build_dss_commands


def test_dss_translation_contains_all_elements():
    net = build_cigre_lv()
    commands = build_dss_commands(net)
    assert sum(c.startswith("New Line.L") for c in commands) == len(net.line)
    assert sum(c.startswith("New Transformer.T") for c in commands) == len(net.trafo)
    assert sum(c.startswith("New Load.Load_") for c in commands) == len(net.load)
