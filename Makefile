PYTHON=.venv/Scripts/python.exe

test:
	$(PYTHON) -m pytest -q

pilot:
	$(PYTHON) scripts/run_data_audit.py

cigre:
	$(PYTHON) scripts/run_cigre_base.py

