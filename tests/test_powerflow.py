from src.powerflow.cigre import compare_algorithms


def test_cigre_algorithms_converge_and_agree():
    result = compare_algorithms()
    assert result["nr"]["converged"] and result["bfsw"]["converged"]
    assert result["cross_algorithm"]["maximum_absolute_voltage_difference_pu"] < 1e-6

