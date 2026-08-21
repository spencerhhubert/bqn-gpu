.PHONY: build-cbqn conformance test test-cpu test-cuda

PYTHON ?= python3

build-cbqn:
	./scripts/build_cbqn.sh

conformance:
	$(PYTHON) scripts/render_conformance.py --check

test: test-cpu

test-cpu: build-cbqn conformance
	BQN_GPU_TEST_DEVICE=CPU $(PYTHON) -m pytest

test-cuda: build-cbqn conformance
	BQN_GPU_TEST_DEVICE=CUDA $(PYTHON) -m pytest
