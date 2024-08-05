PYTHON ?= python

.PHONY: test
test:
	pytest -v

.PHONY: autoformat
autoformat:
	black mazedyn
	isort mazedyn 

.PHONY: lint
lint:
	$(PYTHON) -m flake8 mazedyn
	$(PYTHON) -m black mazedyn --check
	# Note that Bandit will look for .bandit file only if it's invoked with -r option.
	$(PYTHON) -m bandit -c pyproject.toml -r mazedyn --exit-zero
	$(PYTHON) -m mypy --install-types --non-interactive

.PHONY: clean
clean:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: rmtmp
rmtmp:
	sudo rm -r /tmp && sudo mkdir /tmp && sudo chmod -R 777 /tmp

.PHONY: rmprocessed
rmprocessed:
	rm -rf data/processed/*

.PHONY: rmoutputs
rmoutputs:
	rm -rf outputs/*

.PHONY: conda-osx-64.lock
conda-osx-64.lock:
	CONDA_SUBDIR=osx-64 conda-lock -f conda.yaml -p osx-64
	CONDA_SUBDIR=osx-64 conda-lock render -p osx-64

.PHONY: conda-linux-64.lock
conda-linux-64.lock:
	conda-lock -f conda.yaml -p linux-64
	conda-lock render -p linux-64

conda-lock.yml: conda-osx-64.lock conda-linux-64.lock

.PHONY: conda-osx-arm.lock
conda-osx-arm.lock:
	CONDA_SUBDIR=osx-arm64 conda-lock -f conda.yaml -p osx-arm64
	CONDA_SUBDIR=osx-arm64 conda-lock render -p osx-arm64

# Clear the cache and rebuild the lock.
.PHONY: poetry.lock
poetry.lock:
	poetry cache clear --all .
	poetry lock