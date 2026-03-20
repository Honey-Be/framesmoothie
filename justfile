set positional-arguments

python := "python"
pytest := "pytest"
artifacts_dir := "artifacts"
toy_dataset := artifacts_dir + "/toy_panoptic.pt"

default:
    @just --list

compile:
    {{python}} -m py_compile $(find src/framesmoothie -name '*.py')

test-fast:
    {{pytest}} -q tests/test_specs.py tests/test_diag_meter.py tests/test_zone_losses.py tests/test_matcher.py

test-s9:
    {{pytest}} -q tests/test_zoning.py tests/test_hmc_boundary.py tests/test_model_smoke.py

test-gradcheck:
    {{pytest}} -q tests/test_gradcheck_fmlm.py tests/test_smoke_train.py

test-gradcheck-slow:
    {{pytest}} -q tests/test_gradcheck_fmlm.py tests/test_smoke_train.py -m slow

smoke-train:
    {{pytest}} -q tests/test_smoke_train.py -m slow

toy-data:
    mkdir -p {{artifacts_dir}}
    {{python}} make_toy_panoptic.py --output {{toy_dataset}} --num-samples 16 --target-style-shift

list-presets:
    {{python}} -c "from .presets import list_presets; print('\\n'.join(sorted(list_presets())))"

tiny-overfit device="cpu": toy-data
    {{python}} run_tiny_overfit.py --dataset {{toy_dataset}} --steps 100 --device {{device}}

tiny-overfit-preset preset="pred" device="cpu" steps="100" topk="1" early_stop="" early_metric="" early_mode="" early_delta="" ema_schedule="constant" ema_beta_start="0.99" ema_beta_end="0.999": toy-data
    {{python}} run_tiny_overfit.py --dataset {{toy_dataset}} --steps {{steps}} --device {{device}} --preset {{preset}} --save-checkpoints --topk {{topk}} {{ if early_stop != "" { "--early-stop-patience " + early_stop } else { "" } }} {{ if early_metric != "" { "--early-stop-metric " + early_metric } else { "" } }} {{ if early_mode != "" { "--early-stop-mode " + early_mode } else { "" } }} {{ if early_delta != "" { "--early-stop-min-delta " + early_delta } else { "" } }} --ema-schedule-kind {{ema_schedule}} --ema-beta-start {{ema_beta_start}} --ema-beta-end {{ema_beta_end}}

tiny-overfit-suite device="cpu" steps="100" topk="1" early_stop="" early_metric="" early_mode="" early_delta="" ema_schedule="constant" ema_beta_start="0.99" ema_beta_end="0.999": toy-data
    {{python}} run_preset_suite.py --dataset {{toy_dataset}} --steps {{steps}} --device {{device}} --presets baseline zoning pred full full_lite_instance_friendly full_lite_target_instance_friendly full_lite_target_instance_friendly_v3 full_lite_target_instance_friendly_v4 --save-checkpoints --topk {{topk}} {{ if early_stop != "" { "--early-stop-patience " + early_stop } else { "" } }} {{ if early_metric != "" { "--early-stop-metric " + early_metric } else { "" } }} {{ if early_mode != "" { "--early-stop-mode " + early_mode } else { "" } }} {{ if early_delta != "" { "--early-stop-min-delta " + early_delta } else { "" } }} --ema-schedule-kind {{ema_schedule}} --ema-beta-start {{ema_beta_start}} --ema-beta-end {{ema_beta_end}}

plots device="cpu": tiny-overfit-suite {{device}}

diagnostics:
    @echo "See docs/TESTING_GUIDE.md and docs/DIAGNOSTICS_VISUALIZATION.md"

plot-suite suite_dir:
    {{python}} plot_suite_summary.py --suite-dir {{suite_dir}}


tiny-overfit-resume resume preset="pred" device="cpu" steps="100" topk="1" early_stop="" early_metric="" early_mode="" early_delta="" ema_schedule="constant" ema_beta_start="0.99" ema_beta_end="0.999":
    {{python}} run_tiny_overfit.py --dataset {{toy_dataset}} --preset {{preset}} --steps {{steps}} --device {{device}} --resume-from {{resume}} --save-checkpoints --topk {{topk}} {{ if early_stop != "" { "--early-stop-patience " + early_stop } else { "" } }} {{ if early_metric != "" { "--early-stop-metric " + early_metric } else { "" } }} {{ if early_mode != "" { "--early-stop-mode " + early_mode } else { "" } }} {{ if early_delta != "" { "--early-stop-min-delta " + early_delta } else { "" } }} --ema-schedule-kind {{ema_schedule}} --ema-beta-start {{ema_beta_start}} --ema-beta-end {{ema_beta_end}}
