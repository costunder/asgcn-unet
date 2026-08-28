PYTHON ?= .venv/bin/python

.PHONY: setup doctor test lint inspect train

setup:
	bash scripts/setup.sh

doctor:
	$(PYTHON) scripts/check_env.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

inspect:
	$(PYTHON) -m asgcn_recon.cli inspect --config configs/hdr_train.json

train:
	$(PYTHON) -m asgcn_recon.cli train --config configs/hdr_train.json
