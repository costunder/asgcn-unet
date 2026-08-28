PYTHON ?= .venv/bin/python

.PHONY: setup doctor test lint smoke inspect train

setup:
	bash scripts/setup_server.sh

doctor:
	$(PYTHON) scripts/check_environment.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

smoke:
	$(PYTHON) -m asgcn_recon.smoke --workspace data/smoke

inspect:
	$(PYTHON) -m asgcn_recon.cli inspect --config configs/eventhdr_train.json

train:
	$(PYTHON) -m asgcn_recon.cli train --config configs/eventhdr_train.json
