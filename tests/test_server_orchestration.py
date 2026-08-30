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
    all_stage = script.split('  all)\n', maxsplit=1)[1]
    assert all_stage.index("run_check") < all_stage.index("run_profile")
    assert all_stage.index("run_profile") < all_stage.index("run_train")
    assert all_stage.index("run_train") < all_stage.index("run_calibrate")
    assert all_stage.index("run_calibrate") < all_stage.index("run_eval")

    for stage in ("check", "profile", "train", "calibrate", "eval", "all"):
        assert stage in script
    for config in ("configs/train.json", "configs/hdr.json", "configs/aid.json"):
        assert config in script
    assert 'SIMULATION_STEPS_LIST="${SIMULATION_STEPS_LIST:-4 8 16 32}"' in script
    assert "for dynamics in literal_eq15 standard_if" in script
    assert "RUN_BENCHMARK=1" in script
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
    assert 'for config_path in "${HDR_CONFIG}" "${AID_CONFIG}"' in script
    assert "silently skipped" in script
    assert "rm -f" not in script
    assert "write_stage_status" in script
    assert "RUNNING" in script
    assert "COMPLETED" in script
    assert "FAILED" in script
    assert "record_stage_failure" in script


def test_only_three_configs_define_short_output_roots_and_spline_chunking() -> None:
    config_dir = ROOT / "configs"
    assert {path.name for path in config_dir.glob("*.json")} == {
        "train.json",
        "hdr.json",
        "aid.json",
    }

    configs = {
        name: json.loads((config_dir / name).read_text(encoding="utf-8"))
        for name in ("train.json", "hdr.json", "aid.json")
    }
    assert configs["train.json"]["output"]["run_dir"] == "runs/train"
    assert configs["hdr.json"]["eval"]["output_dir"] == "runs/eval/hdr"
    assert configs["aid.json"]["eval"]["output_dir"] == "runs/eval/aid"
    assert configs["train.json"]["model"] == configs["hdr.json"]["model"]
    assert configs["train.json"]["model"] == configs["aid.json"]["model"]
    for config in configs.values():
        assert config["model"]["spline_chunk_size"] == 65536


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
    assert 'PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-runs/profile.json}"' in script
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


def test_readme_starts_with_public_https_quickstart_and_manual_private_restoration() -> None:
    readme = _text("README.md")
    quickstart_heading = "## 빠른 시작 (MobaXterm)\n"
    assert readme.index(quickstart_heading) < readme.index("## 구현 범위와 모델 구조")
    quickstart = readme.split(quickstart_heading, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]

    clone_command = (
        "cd ~\n"
        "git clone https://github.com/costunder/asgcn-unet.git &&\n"
        "cd asgcn-unet"
    )
    conda_command = "conda create -n asgcn --override-channels -c conda-forge python=3.12 git"
    assert clone_command in quickstart
    assert conda_command in quickstart
    assert quickstart.index(clone_command) < quickstart.index(conda_command)
    assert "conda activate asgcn" in quickstart
    public_clone = quickstart.split("### 1.", maxsplit=1)[1].split("\n### ", maxsplit=1)[0]
    assert "Public" in public_clone
    for line in quickstart.splitlines():
        if line.startswith("conda create "):
            assert "gh" not in line.split()
            assert " -y " not in f" {line} "
            assert "--yes" not in line

    assert "prepare_asgcn_deploy_key" not in readme
    for private_auth_command in ("gh auth", "ssh-keygen", "git@"):
        assert private_auth_command not in quickstart
    for absolute_home_path in ("/home/", "/Users/", "C:\\Users\\"):
        assert absolute_home_path not in quickstart
    assert "ASGCN_DIR" not in quickstart
    assert "bash scripts/setup.sh" in quickstart
    assert "source .venv/bin/activate" in quickstart
    assert "python scripts/check_env.py --require-cuda --lock constraints/py312.txt" in quickstart
    assert "bash scripts/get_aid.sh --all" in quickstart
    assert "bash scripts/get_hdr.sh --download" in quickstart
    assert "bash scripts/get_hdr.sh --archive" not in quickstart
    assert "mkdir -p data/_archives" not in quickstart
    assert "data/EventHDR/{train,eval}" in quickstart
    assert "SHA-256" in quickstart
    assert "python scripts/check_env.py --require-full-data --lock constraints/py312.txt" in quickstart
    assert "bash scripts/run.sh all" in quickstart

    private_restoration = quickstart.split("### 5.", maxsplit=1)[1]
    for visibility_label in (
        "Private",
        "Settings",
        "Danger Zone",
        "Change repository visibility",
        "Make this repository private",
    ):
        assert visibility_label in private_restoration
    assert "수동" in private_restoration or "직접" in private_restoration
    assert "실험 완료를 감시하거나 자동으로 Private으로 바꾸는 기능은 없다." in private_restoration
    assert "docs/SERVER.md#1-public-저장소-clone과-설치" in quickstart
