PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: venv install test demo serve clean

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -e ".[server,dev]"

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m n4c_decision_broker.cli demo

serve:
	$(PY) -m n4c_decision_broker.cli serve

clean:
	rm -rf .pytest_cache dist build *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
