.PHONY: benchmark build-cbqn conformance corpus test test-cpu test-cuda

PYTHON ?= python3

build-cbqn:
	./scripts/build_cbqn.sh

conformance:
	$(PYTHON) scripts/render_conformance.py --check

corpus:
	$(PYTHON) scripts/generate_corpus.py --check

test: test-cpu

test-cpu: build-cbqn conformance corpus
	BQN_GPU_TEST_DEVICE=CPU $(PYTHON) -m pytest

test-cuda: build-cbqn conformance corpus
	BQN_GPU_TEST_DEVICE=CUDA $(PYTHON) -m pytest

benchmark: build-cbqn
	$(PYTHON) scripts/run_corpus.py --backend cbqn --backend tinygrad --backend torch
