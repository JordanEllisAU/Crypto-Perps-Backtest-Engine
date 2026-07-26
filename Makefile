# Crypto-Perps-Backtest-Engine — local CI gate.

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: lint test ci clean

lint:
	$(PYTHON) -m compileall -q src scripts
	$(PYTHON) scripts/slop_sentinel.py

test:
	$(PYTHON) -m pytest -q

ci: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
