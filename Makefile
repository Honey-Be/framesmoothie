PYTHON ?= poetry run python
PYTEST ?= poetry run pytest
ARTIFACTS_DIR ?= artifacts
TOY_DATASET ?= $(ARTIFACTS_DIR)/toy_panoptic.pt
SMOKE_DEVICE ?= cpu

.PHONY: help compile test-fast test-s9 test-slow smoke-train toy-data list-presets tiny-overfit tiny-overfit-preset tiny-overfit-suite tiny-overfit-resume diagnostics plots all

help:
	@echo "Available targets:"
	@echo "  compile       - py_compile over src/framesmoothie"
	@echo "  test-fast     - quick unit tests (no s9 required)"
	@echo "  test-s9       - zoning/HMC/model smoke tests (requires s9)"
	@echo "  test-slow     - gradcheck + smoke train (requires s9)"
	@echo "  toy-data      - build toy panoptic dataset into $(TOY_DATASET)"
	@echo "  list-presets  - show available tiny-overfit presets"
	@echo "  tiny-overfit  - run tiny overfit with default preset on $(TOY_DATASET)"
	@echo "  tiny-overfit-preset - run tiny overfit with PRESET=<name>"
	@echo "  tiny-overfit-suite  - run a preset suite and aggregate summaries"
	@echo "  tiny-overfit-resume - resume a run from RESUME=<checkpoint.safetensors>"
	@echo "  vars: TOPK=<k> EARLY_STOP=<patience> EARLY_METRIC=<metric> SCHEDULER=<none|cosine|step> SCHED_STEP=<n> SCHED_GAMMA=<g> EMA_SCHEDULER=<constant|linear|cosine> EMA_BETA_START=<b0> EMA_BETA_END=<b1>"
	@echo "  plots         - generate suite plots from SUITE_DIR"
	@echo "  diagnostics   - print docs locations"
	@echo "  all           - compile + test-fast"

compile:
	$(PYTHON) -m py_compile $$(find src/framesmoothie -name '*.py')

test-fast:
	$(PYTEST) -q tests/test_specs.py tests/test_diag_meter.py tests/test_zone_losses.py tests/test_matcher.py

test-s9:
	$(PYTEST) -q tests/test_zoning.py tests/test_hmc_boundary.py tests/test_model_smoke.py

test-gradcheck:
	$(PYTEST) -q tests/test_gradcheck_fmlm.py tests/test_smoke_train.py

test-gradcheck-slow:
	$(PYTEST) -q tests/test_gradcheck_fmlm.py tests/test_smoke_train.py -m slow

smoke-train:
	$(PYTEST) -q tests/test_smoke_train.py -m slow

toy-data:
	mkdir -p $(ARTIFACTS_DIR)
	$(PYTHON) -m framesmoothie._scripts.make_toy_panoptic --output $(TOY_DATASET) --num-samples 16 --target-style-shift


list-presets:
	$(PYTHON) -c "from framesmoothie._scripts.presets import list_presets; print('\\n'.join(sorted(list_presets())))"

tiny-overfit-preset: toy-data
	$(PYTHON) -m framesmoothie._scripts.run_tiny_overfit --dataset $(TOY_DATASET) --steps $${STEPS:-100} --device $(SMOKE_DEVICE) --preset $${PRESET:-pred} --save-checkpoints --topk $${TOPK:-1} $${EARLY_STOP:+--early-stop-patience $${EARLY_STOP}} $${EARLY_METRIC:+--early-stop-metric $${EARLY_METRIC}} $${EARLY_MODE:+--early-stop-mode $${EARLY_MODE}} $${EARLY_DELTA:+--early-stop-min-delta $${EARLY_DELTA}} $${SCHEDULER:+--scheduler-kind $${SCHEDULER}} $${SCHED_STEP:+--scheduler-step-size $${SCHED_STEP}} $${SCHED_GAMMA:+--scheduler-gamma $${SCHED_GAMMA}} $${EMA_SCHEDULER:+--ema-schedule-kind $${EMA_SCHEDULER}} $${EMA_BETA_START:+--ema-beta-start $${EMA_BETA_START}} $${EMA_BETA_END:+--ema-beta-end $${EMA_BETA_END}}

tiny-overfit-suite: toy-data
	$(PYTHON) -m framesmoothie._scripts.run_preset_suite --dataset $(TOY_DATASET) --steps $${STEPS:-100} --device $(SMOKE_DEVICE) --presets baseline zoning pred full full_lite_instance_friendly full_lite_target_instance_friendly full_lite_target_instance_friendly_v3 full_lite_target_instance_friendly_v4 --save-checkpoints --topk $${TOPK:-1} $${EARLY_STOP:+--early-stop-patience $${EARLY_STOP}} $${EARLY_METRIC:+--early-stop-metric $${EARLY_METRIC}} $${EARLY_MODE:+--early-stop-mode $${EARLY_MODE}} $${EARLY_DELTA:+--early-stop-min-delta $${EARLY_DELTA}} $${SCHEDULER:+--scheduler-kind $${SCHEDULER}} $${SCHED_STEP:+--scheduler-step-size $${SCHED_STEP}} $${SCHED_GAMMA:+--scheduler-gamma $${SCHED_GAMMA}} $${EMA_SCHEDULER:+--ema-schedule-kind $${EMA_SCHEDULER}} $${EMA_BETA_START:+--ema-beta-start $${EMA_BETA_START}} $${EMA_BETA_END:+--ema-beta-end $${EMA_BETA_END}}

plots:
	@if [ -z "$(SUITE_DIR)" ]; then echo "Usage: make plots SUITE_DIR=artifacts/suites/<suite_name>"; exit 1; fi
	$(PYTHON) -m framesmoothie._scripts.plot_suite_summary --suite-dir $(SUITE_DIR)

tiny-overfit: toy-data
	$(PYTHON) -m framesmoothie._scripts.run_tiny_overfit --dataset $(TOY_DATASET) --steps 100 --device $(SMOKE_DEVICE)

diagnostics:
	@echo "See docs/TESTING_GUIDE.md and docs/DIAGNOSTICS_VISUALIZATION.md"

all: compile test-fast


tiny-overfit-resume:
	@if [ -z "$${RESUME}" ]; then echo "Usage: make tiny-overfit-resume RESUME=artifacts/runs/<run>/checkpoints/step_000100.safetensors"; exit 1; fi
	$(PYTHON) -m framesmoothie._scripts.run_tiny_overfit --dataset $(TOY_DATASET) --steps $${STEPS:-100} --device $(SMOKE_DEVICE) --resume-from $${RESUME} --save-checkpoints $${TOPK:+--topk $${TOPK}} $${EARLY_STOP:+--early-stop-patience $${EARLY_STOP}} $${EARLY_METRIC:+--early-stop-metric $${EARLY_METRIC}} $${EARLY_MODE:+--early-stop-mode $${EARLY_MODE}} $${EARLY_DELTA:+--early-stop-min-delta $${EARLY_DELTA}} $${SCHEDULER:+--scheduler-kind $${SCHEDULER}} $${SCHED_STEP:+--scheduler-step-size $${SCHED_STEP}} $${SCHED_GAMMA:+--scheduler-gamma $${SCHED_GAMMA}} $${EMA_SCHEDULER:+--ema-schedule-kind $${EMA_SCHEDULER}} $${EMA_BETA_START:+--ema-beta-start $${EMA_BETA_START}} $${EMA_BETA_END:+--ema-beta-end $${EMA_BETA_END}}
