PYTHON ?= .venv/bin/python

.PHONY: setup data doctor test lint inspect train full

setup:
	bash scripts/setup.sh

data:
	bash scripts/get_aid.sh --all

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

full:
	bash scripts/full.sh
