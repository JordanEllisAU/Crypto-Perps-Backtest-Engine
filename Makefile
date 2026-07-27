# Crypto-Perps-Backtest-Engine — local CI gate.

PYTHON ?= $(shell [ -d .venv ] && echo .venv/bin/python || echo python)

.PHONY: lint test ci clean

lint:
	$(PYTHON) -m compileall -q engine_core src scripts
	$(PYTHON) scripts/slop_sentinel.py

test:
	$(PYTHON) -m pytest -q

ci: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
