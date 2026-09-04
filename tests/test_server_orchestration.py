from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_run_script_exposes_restartable_ordered_stages() -> None:
    script = _text("scripts/run.sh")

    assert "[check]" in script
    assert "[profile]" in script
    assert "[train]" in script
    assert "[calibrate]" in script
    assert "[eval]" in script
    assert "[eval-hdr]" in script
    assert "[eval-aid]" in script
    all_stage = script.split('  all)\n', maxsplit=1)[1]
    assert all_stage.index("run_check") < all_stage.index("run_profile")
    assert all_stage.index("run_profile") < all_stage.index("run_train")
    assert all_stage.index("run_train") < all_stage.index("run_calibrate")
    assert all_stage.index("run_calibrate") < all_stage.index("run_eval")

    for stage in (
        "check", "profile", "train", "calibrate", "eval", "eval-hdr", "eval-aid", "all"
    ):
        assert stage in script
    for config in ("configs/train.json", "configs/hdr.json", "configs/aid.json"):
        assert config in script
    assert 'SIMULATION_STEPS_LIST="${SIMULATION_STEPS_LIST:-4 8 16 32}"' in script
    assert "for dynamics in literal_eq15 standard_if" in script
    assert 'RUN_BENCHMARK="${run_benchmark}"' in script
    assert 'CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"' in script
    assert 'PROFILE_SAMPLES="${PROFILE_SAMPLES:-3}"' in script
    assert 'PROFILE_TOP_DENSITY="${PROFILE_TOP_DENSITY:-10}"' in script
    assert "asgcn_unet.cli profile" in script
    assert "all is a reporting pipeline and cannot bypass" in script
    assert "profile/all is a CUDA reporting gate" in script
    assert "PROFILE_TOP_DENSITY must be >= PROFILE_SAMPLES" in script
    assert "including PROFILE_OUTPUT" in script
    assert "\\$PROJECT_ROOT" in script
    assert 'for config_path in "${TRAIN_CONFIG}" "${AID_CONFIG}"' in script
    assert 'run_eval_config "${HDR_CONFIG}"' in script
    assert 'run_eval_config "${AID_CONFIG}"' in script
    assert "only with explicit EVAL_RESUME=1" in script
    assert "rm -f" not in script
    assert "write_stage_status" in script
    assert "RUNNING" in script
    assert "COMPLETED" in script
    assert "FAILED" in script
    assert "record_stage_failure" in script
    assert "execute_stage eval-hdr run_eval_hdr" in script
    assert "execute_stage eval-aid run_eval_aid" in script


def test_baseline_and_batch_configs_keep_short_separate_output_roots() -> None:
    config_dir = ROOT / "configs"
    assert {path.name for path in config_dir.glob("*.json")} == {
        "train.json",
        "batch.json",
        "fast.json",
        "hdr.json",
        "hdr-fast.json",
        "aid.json",
        "aid-fast.json",
    }

    configs = {
        name: json.loads((config_dir / name).read_text(encoding="utf-8"))
        for name in (
            "train.json", "batch.json", "fast.json", "hdr.json", "hdr-fast.json",
            "aid.json", "aid-fast.json",
        )
    }
    assert configs["train.json"]["output"]["run_dir"] == "runs/train"
    assert configs["batch.json"]["output"]["run_dir"] == "runs/batch"
    assert configs["batch.json"]["train"]["batch_size"] == 4
    assert configs["batch.json"]["train"]["batching"] == "independent_sequences"
    assert configs["batch.json"]["train"]["timing_steps"] == 50
    assert configs["batch.json"]["dataset"] == configs["train.json"]["dataset"]
    assert configs["batch.json"]["model"] == configs["train.json"]["model"]
    for key, value in configs["train.json"]["train"].items():
        if key != "batch_size":
            assert configs["batch.json"]["train"][key] == value, key
    expected_fast = json.loads(json.dumps(configs["batch.json"]))
    expected_fast["train"]["batch_size"] = 16
    expected_fast["model"]["spline_backend"] = "triton"
    expected_fast["output"]["run_dir"] = "runs/fast"
    assert configs["fast.json"] == expected_fast
    assert configs["hdr.json"]["eval"]["output_dir"] == "runs/eval/hdr"
    assert configs["aid.json"]["eval"]["output_dir"] == "runs/eval/aid"
    assert configs["train.json"]["model"] == configs["hdr.json"]["model"]
    assert configs["train.json"]["model"] == configs["aid.json"]["model"]
    assert configs["fast.json"]["model"] == configs["hdr-fast.json"]["model"]
    assert configs["fast.json"]["model"] == configs["aid-fast.json"]["model"]
    assert configs["hdr-fast.json"]["eval"]["output_dir"] == "runs/fast/eval/hdr"
    assert configs["aid-fast.json"]["eval"]["output_dir"] == "runs/fast/eval/aid"
    for config in configs.values():
        assert config["model"]["spline_chunk_size"] == 65536


def test_batch_wrapper_isolates_training_profile_status_and_evaluation() -> None:
    script = _text("scripts/run.sh")
    assert 'EXPERIMENT="${EXPERIMENT:-single}"' in script
    for assignment in (
        "DEFAULT_TRAIN_CONFIG=configs/batch.json",
        "DEFAULT_TRAIN_RUN=runs/batch",
        "DEFAULT_PROFILE_OUTPUT=runs/batch-profile.json",
        "DEFAULT_STATUS_DIR=runs/batch-status",
        "DEFAULT_EVAL_ROOT=runs/batch/eval",
    ):
        assert assignment in script
    assert 'EVAL_OUTPUT_DIR="${output_dir}"' in script
    evaluation = _text("scripts/eval.sh")
    assert 'OUTPUT_ARGS=(--output-dir "${EVAL_OUTPUT_DIR}")' in evaluation
    assert evaluation.count('"${OUTPUT_ARGS[@]}"') == 2


def test_evaluation_recovery_routes_one_dataset_and_forwards_edge_guard() -> None:
    runner = _text("scripts/run.sh")
    evaluation = _text("scripts/eval.sh")

    assert 'EVAL_MAX_GRAPH_EDGES="${EVAL_MAX_GRAPH_EDGES:-}"' in runner
    assert 'EVAL_MAX_GRAPH_EDGES="${EVAL_MAX_GRAPH_EDGES}"' in runner
    assert 'run_eval_config "${HDR_CONFIG}"' in runner
    assert 'run_eval_config "${AID_CONFIG}"' in runner
    assert '${STATUS_DIR}/${stage_name}.json' in runner

    assert 'EVAL_MAX_GRAPH_EDGES="${EVAL_MAX_GRAPH_EDGES:-}"' in evaluation
    assert 'EVAL_MAX_GRAPH_EDGES must be a positive integer' in evaluation
    assert 'GRAPH_EDGE_ARGS=(--max-graph-edges-override "${EVAL_MAX_GRAPH_EDGES}")' in evaluation
    assert evaluation.count('"${GRAPH_EDGE_ARGS[@]}"') == 2


def test_evaluation_mode_resume_is_explicit_and_preserves_partial_artifacts() -> None:
    runner = _text("scripts/run.sh")
    evaluation = _text("scripts/eval.sh")

    assert 'EVAL_RESUME="${EVAL_RESUME:-0}"' in runner
    assert "EVAL_RESUME=1 requires an explicit EVAL_OUTPUT_ROOT" in runner
    assert "quality complete; running benchmark only" in runner
    assert "benchmark complete; restarting quality only" in runner
    assert "restarting incomplete mode" in runner
    assert 'RUN_EVALUATION="${run_evaluation}"' in runner
    assert 'RUN_BENCHMARK="${run_benchmark}"' in runner
    assert 'mv -- "${path}" "${backup}"' in runner
    assert 'flock --nonblock "${resume_lock_fd}"' in runner
    assert '( -e "${run_dir}" || -L "${run_dir}" )' in runner
    assert "rm -f" not in runner

    assert 'RUN_EVALUATION="${RUN_EVALUATION:-1}"' in evaluation
    assert "RUN_EVALUATION and RUN_BENCHMARK cannot both be 0" in evaluation
    assert 'if [[ "${RUN_EVALUATION}" == "1" ]]' in evaluation
    assert "Skipping completed quality evaluation" in evaluation


def test_fast_wrapper_isolates_measured_training_and_safe_pause() -> None:
    script = _text("scripts/run.sh")
    for assignment in (
        "DEFAULT_TRAIN_CONFIG=configs/fast.json",
        "DEFAULT_TRAIN_RUN=runs/fast",
        "DEFAULT_PROFILE_OUTPUT=runs/fast-profile.json",
        "DEFAULT_STATUS_DIR=runs/fast-status",
        "DEFAULT_EVAL_ROOT=runs/fast/eval",
        "DEFAULT_HDR_CONFIG=configs/hdr-fast.json",
        "DEFAULT_AID_CONFIG=configs/aid-fast.json",
    ):
        assert assignment in script
    assert '[[ "${ACTIVE_STAGE}" == "train" && "${exit_code}" == "75" ]]' in script
    assert 'write_stage_status "${ACTIVE_STAGE}" PAUSED' in script
    training = _text("scripts/train.sh")
    assert 'TRAIN_ARGS+=(--max-hours "${MAX_HOURS}")' in training
    assert 'TRAIN_ARGS+=(--checkpoint-seconds "${CHECKPOINT_SECONDS}")' in training


def test_calibration_wrapper_defaults_to_all_samples_and_protects_output() -> None:
    script = _text("scripts/calibrate.sh")
    assert 'CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"' in script
    assert 'OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"' in script
    assert '--samples "${CALIBRATION_SAMPLES}"' in script
    assert "calibrated output already exists" in script
    assert "CALIBRATE_ARGS+=(--overwrite)" in script
    assert "rm -f" not in script


def test_all_wrappers_support_optional_validate_all_preflight() -> None:
    for relative in ("scripts/train.sh", "scripts/eval.sh", "scripts/calibrate.sh"):
        script = _text(relative)
        assert "INSPECT_VALIDATE_ALL" in script
        assert "INSPECT_ARGS+=(--validate-all)" in script


def test_training_wrapper_requires_verified_profile_or_explicit_nonreporting_bypass() -> None:
    script = _text("scripts/train.sh")
    assert 'PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${PROFILE_OUTPUT:-${DEFAULT_PROFILE_OUTPUT}}}"' in script
    assert 'ALLOW_UNVERIFIED_PREFLIGHT="${ALLOW_UNVERIFIED_PREFLIGHT:-0}"' in script
    assert "--preflight-report" in script
    assert "--allow-unverified-preflight" in script
    assert "non-reporting run" in script


def test_profile_has_slurm_and_pbs_entrypoints_and_train_dependency() -> None:
    for relative in ("server/profile.sbatch", "server/profile.pbs"):
        script = _text(relative)
        assert script.startswith("#!/usr/bin/env bash\n")
        assert "scripts/run.sh" in script
        assert "profile" in script
        assert "PROFILE_OUTPUT" in script
        assert "afterok" in script


def test_calibration_has_slurm_and_pbs_entrypoints_with_dependency_examples() -> None:
    for relative in ("server/calibrate.sbatch", "server/calibrate.pbs"):
        script = _text(relative)
        assert script.startswith("#!/usr/bin/env bash\n")
        assert "scripts/calibrate.sh" in script
        assert "CALIBRATION_SAMPLES" in script
        assert "depend" in script


def test_all_scheduler_entrypoints_use_short_paths_and_optional_cuda_module() -> None:
    for scheduler in ("sbatch", "pbs"):
        for stage in ("train", "calibrate", "eval"):
            script = _text(f"server/{stage}.{scheduler}")
            assert "CUDA_MODULE" in script
            assert "module load" in script
            assert "runs/eventhdr_asgcn" not in script
            assert "hdr_train.json" not in script
            assert "hdr_ann.json" not in script
            assert "hdr_snn.json" not in script


def test_scheduler_fast_routing_keeps_profile_checkpoints_and_eval_isolated() -> None:
    for scheduler in ("sbatch", "pbs"):
        for stage in ("profile", "train", "calibrate", "eval"):
            script = _text(f"server/{stage}.{scheduler}")
            assert 'EXPERIMENT="${EXPERIMENT:-single}"' in script
            assert "configs/fast.json" in script or "configs/hdr-fast.json" in script
        training = _text(f"server/train.{scheduler}")
        assert 'MAX_HOURS="${MAX_HOURS:-47}"' in training
        assert 'CHECKPOINT_SECONDS="${CHECKPOINT_SECONDS:-300}"' in training
        evaluation = _text(f"server/eval.{scheduler}")
        assert "DEFAULT_EVAL_ROOT=runs/batch/eval" in evaluation
        assert 'EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${DEFAULT_EVAL_ROOT}/${DEFAULT_DATASET}}"' in evaluation


def test_scheduler_logs_require_explicit_opt_in_for_private_provenance() -> None:
    path_fields = {
        "profile": ("TRAIN_CONFIG", "PROFILE_OUTPUT"),
        "train": ("CONFIG_PATH", "RESUME_CHECKPOINT"),
        "calibrate": ("CONFIG_PATH", "CHECKPOINT_PATH", "OUTPUT_PATH"),
        "eval": ("CONFIG_PATH", "CHECKPOINT_PATH"),
    }
    marker = 'if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then\n'
    for scheduler in ("sbatch", "pbs"):
        for stage, fields in path_fields.items():
            script = _text(f"server/{stage}.{scheduler}")
            assert 'INCLUDE_PRIVATE_HOST_PROVENANCE="${INCLUDE_PRIVATE_HOST_PROVENANCE:-0}"' in script
            assert marker in script
            private_and_public = script.split(marker, maxsplit=1)[1]
            private_block, public_tail = private_and_public.split("\nelse\n", maxsplit=1)
            public_block = public_tail.split("\nfi", maxsplit=1)[0]

            assert "$(hostname)" in private_block
            job_variable = "SLURM_JOB_ID" if scheduler == "sbatch" else "PBS_JOBID"
            assert job_variable in private_block
            assert "$(hostname)" not in public_block
            assert job_variable not in public_block
            for field in fields:
                assert f"${{{field}}}" in private_block
                assert f"${{{field}##*/}}" in public_block


def test_shell_wrappers_default_to_portable_path_labels() -> None:
    for relative in (
        "scripts/run.sh",
        "scripts/train.sh",
        "scripts/calibrate.sh",
        "scripts/eval.sh",
    ):
        script = _text(relative)
        assert 'INCLUDE_PRIVATE_HOST_PROVENANCE="${INCLUDE_PRIVATE_HOST_PROVENANCE:-0}"' in script
        assert 'if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then' in script
        assert 'printf \'%s\' "${path##*/}"' in script


def test_scheduler_docs_do_not_export_the_complete_login_environment() -> None:
    for relative in (
        "README.md",
        "docs/SERVER.md",
        "hand_off.md",
        "server/calibrate.sbatch",
        "server/eval.sbatch",
        "server/train.sbatch",
    ):
        assert "--export=ALL" not in _text(relative)

    readme = _text("README.md")
    assert '--export=PROJECT_ROOT="$PWD"' in readme
    assert 'CONDA_PREFIX="$CONDA_PREFIX"' in readme
    assert "INCLUDE_PRIVATE_HOST_PROVENANCE=1" in readme


def test_scheduler_docs_sanitize_log_names_and_contents_before_publication() -> None:
    for relative in ("README.md", "docs/SERVER.md", "hand_off.md"):
        documentation = _text(relative)
        assert "파일명 자체에 job ID" in documentation
        assert "<job-name>.o<job-id>" in documentation
        assert "중립 파일명" in documentation
        assert "공개 후보" in documentation
        assert "scripts/scan_private_text.py" in documentation
        assert "--require-external-patterns" in documentation
        assert "INCLUDE_PRIVATE_HOST_PROVENANCE=1" in documentation
        assert "opt-in log" in documentation
        assert "비공개" in documentation

    server_guide = _text("docs/SERVER.md")
    assert 'cp -- "$raw_log" "$public_log"' in server_guide
    assert 'python scripts/scan_private_text.py "$public_log"' in server_guide
    assert "raw 이름을 archive member 이름으로 보존하지 않는다" in server_guide


def test_eventaid_downloader_defaults_to_the_complete_release() -> None:
    script = _text("scripts/get_aid.sh")
    assert "the complete 14-scene release is downloaded" in script
    assert "SCENES=(R-bear)" not in script
    assert 'if ((DOWNLOAD_ALL == 0)) && ((${#SCENES[@]} == 0)); then\n  DOWNLOAD_ALL=1' in script


def test_readme_documents_reproducible_installation_and_full_experiment() -> None:
    readme = _text("README.md")
    installation_heading = "## 설치 및 실행\n"
    installation = readme.split(installation_heading, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]

    clone_command = (
        "git clone https://github.com/costunder/asgcn-unet.git &&\n"
        "cd asgcn-unet"
    )
    conda_command = "conda create -n asgcn --override-channels -c conda-forge python=3.12.14 pip"
    assert clone_command in installation
    assert "cd ~" not in installation
    assert conda_command in installation
    assert installation.index(clone_command) < installation.index(conda_command)
    assert "conda activate asgcn" in installation
    for line in installation.splitlines():
        if line.startswith("conda create "):
            assert "gh" not in line.split()
            assert " -y " not in f" {line} "
            assert "--yes" not in line

    assert "prepare_asgcn_deploy_key" not in readme
    for private_auth_command in ("gh auth", "ssh-keygen", "git@"):
        assert private_auth_command not in installation
    for absolute_home_path in ("/home/", "/Users/", "C:\\Users\\"):
        assert absolute_home_path not in installation
    assert "ASGCN_DIR" not in installation
    assert "conda activate asgcn\nbash scripts/setup.sh" in installation
    assert ".venv" not in installation
    assert ".env.example" not in installation
    assert "constraints/server.json" in installation
    assert "2.13.0+cu126" in installation
    assert "bash scripts/get_aid.sh --all" in installation
    assert "bash scripts/get_hdr.sh --download" in installation
    assert "bash scripts/get_hdr.sh --archive" not in installation
    assert "mkdir -p data/_archives" not in installation
    assert "data/EventHDR/{train,eval}" in installation
    assert "SHA-256" in installation
    assert "python scripts/check_env.py --require-full-data --lock constraints/py312.txt" in installation
    assert "--runtime-profile constraints/server.json" in installation
    assert "bash scripts/run.sh all" in installation
    assert installation.index(conda_command) < installation.index("bash scripts/setup.sh")
    assert installation.index("bash scripts/setup.sh") < installation.index("bash scripts/get_aid.sh")
    assert installation.index("bash scripts/get_hdr.sh") < installation.index("scripts/check_env.py")
    assert installation.index("scripts/check_env.py") < installation.index("bash scripts/run.sh all")
    assert "docs/SERVER.md#1-환경-설치" in installation
    server_guide = _text("docs/SERVER.md")
    assert "## 1. 환경 설치\n" in server_guide
    assert "../README.md#설치-및-실행" in server_guide


def test_research_docs_exclude_repository_visibility_workflows() -> None:
    for relative in ("README.md", "docs/SERVER.md", "hand_off.md"):
        documentation = _text(relative)
        for visibility_instruction in (
            "Change repository visibility",
            "Make this repository private",
            "Danger Zone",
            "setting-repository-visibility",
            "Private 복귀",
            "Private으로 되돌리기",
        ):
            assert visibility_instruction not in documentation, relative
