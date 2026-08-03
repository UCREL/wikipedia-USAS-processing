from pathlib import Path

import pytest
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor

from wikipedia_processing.executors import (
    ExecutorBackend,
    PipelineExecutorFactory,
    SlurmExecutorSettings,
)


def test_executor_backend_values_match_cli_option_strings() -> None:
    assert ExecutorBackend.local == "local"
    assert ExecutorBackend.slurm == "slurm"


def test_slurm_executor_settings_defaults() -> None:
    settings = SlurmExecutorSettings(partition="cpu", time="1:00:00")
    assert settings.cpus_per_task == 1
    assert settings.mem_per_cpu_gb == 2
    assert settings.qos == "normal"
    assert settings.venv_path is None
    assert settings.condaenv is None
    assert settings.mail_user is None
    assert settings.mail_type == "ALL"
    assert settings.sbatch_args is None


def test_pipeline_executor_factory_local_backend_does_not_require_slurm_settings() -> None:
    factory = PipelineExecutorFactory(backend=ExecutorBackend.local, randomize_start_duration=5, skip_completed=True)
    assert factory.slurm_settings is None


def test_pipeline_executor_factory_slurm_backend_raises_value_error_without_slurm_settings() -> None:
    with pytest.raises(ValueError, match="slurm_settings is required"):
        PipelineExecutorFactory(backend=ExecutorBackend.slurm, randomize_start_duration=0, skip_completed=False)


def test_pipeline_executor_factory_slurm_backend_accepts_slurm_settings() -> None:
    settings = SlurmExecutorSettings(partition="cpu", time="1:00:00")
    factory = PipelineExecutorFactory(
        backend=ExecutorBackend.slurm,
        randomize_start_duration=0,
        skip_completed=False,
        slurm_settings=settings,
    )
    assert factory.slurm_settings is settings


def test_create_local_backend_returns_local_pipeline_executor(tmp_path: Path) -> None:
    factory = PipelineExecutorFactory(backend=ExecutorBackend.local, randomize_start_duration=3, skip_completed=True)

    executor = factory.create(pipeline=[], tasks=4, workers=2, logging_dir=str(tmp_path))

    assert isinstance(executor, LocalPipelineExecutor)
    assert executor.tasks == 4
    assert executor.workers == 2
    assert executor.skip_completed is True
    assert executor.randomize_start_duration == 3
    assert executor.logging_dir.path == str(tmp_path)
    assert executor.depends is None


def test_create_local_backend_passes_depends_through(tmp_path: Path) -> None:
    factory = PipelineExecutorFactory(backend=ExecutorBackend.local, randomize_start_duration=0, skip_completed=False)
    upstream = factory.create(pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path / "upstream"))

    downstream = factory.create(
        pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path / "downstream"), depends=upstream
    )

    assert isinstance(downstream, LocalPipelineExecutor)
    assert downstream.depends is upstream


def test_create_local_backend_ignores_job_name(tmp_path: Path) -> None:
    # job_name is documented as Slurm-only; the local backend should accept
    # and silently ignore it rather than erroring.
    factory = PipelineExecutorFactory(backend=ExecutorBackend.local, randomize_start_duration=0, skip_completed=True)

    executor = factory.create(pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path), job_name="reading")

    assert not hasattr(executor, "job_name")


def test_create_local_backend_ignores_tasks_per_job(tmp_path: Path) -> None:
    # tasks_per_job is documented as Slurm-only; the local backend should accept
    # and silently ignore it rather than erroring.
    factory = PipelineExecutorFactory(backend=ExecutorBackend.local, randomize_start_duration=0, skip_completed=True)

    executor = factory.create(pipeline=[], tasks=4, workers=2, logging_dir=str(tmp_path), tasks_per_job=2)

    assert not hasattr(executor, "tasks_per_job")


def _slurm_factory(
    venv_path: Path | None = None,
    condaenv: str | None = None,
    mail_user: str | None = None,
    sbatch_args: dict[str, str] | None = None,
) -> PipelineExecutorFactory:
    settings = SlurmExecutorSettings(
        partition="cpu",
        time="1:00:00",
        venv_path=venv_path,
        condaenv=condaenv,
        mail_user=mail_user,
        sbatch_args=sbatch_args,
    )
    return PipelineExecutorFactory(
        backend=ExecutorBackend.slurm, randomize_start_duration=7, skip_completed=False, slurm_settings=settings
    )


def test_create_slurm_backend_returns_slurm_pipeline_executor(tmp_path: Path) -> None:
    factory = _slurm_factory()

    executor = factory.create(pipeline=[], tasks=4, workers=2, logging_dir=str(tmp_path), job_name="reading")

    assert isinstance(executor, SlurmPipelineExecutor)
    assert executor.tasks == 4
    assert executor.workers == 2
    assert executor.job_name == "reading"
    assert executor.partition == "cpu"
    assert executor.time == "1:00:00"
    assert executor.cpus_per_task == 1
    assert executor.mem_per_cpu_gb == 2
    assert executor.qos == "normal"
    assert executor.mail_type == "ALL"
    assert executor.venv_path is None
    assert executor.condaenv is None
    assert executor.mail_user is None
    assert executor._sbatch_args == {}
    assert executor.skip_completed is False
    assert executor.randomize_start_duration == 7
    assert executor.tasks_per_job == 1


def test_create_slurm_backend_passes_tasks_per_job_through(tmp_path: Path) -> None:
    factory = _slurm_factory()

    executor = factory.create(pipeline=[], tasks=4, workers=2, logging_dir=str(tmp_path), tasks_per_job=2)

    assert isinstance(executor, SlurmPipelineExecutor)
    assert executor.tasks_per_job == 2


def test_create_slurm_backend_defaults_job_name_to_data_processing(tmp_path: Path) -> None:
    factory = _slurm_factory()

    executor = factory.create(pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path))

    assert isinstance(executor, SlurmPipelineExecutor)
    assert executor.job_name == "data_processing"


def test_create_slurm_backend_resolves_venv_path_to_absolute_string(tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    factory = _slurm_factory(venv_path=venv_dir)

    executor = factory.create(pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path / "logs"))

    assert isinstance(executor, SlurmPipelineExecutor)
    assert executor.venv_path == str(venv_dir.resolve())


def test_create_slurm_backend_passes_condaenv_mail_user_and_sbatch_args_job_name_through(tmp_path: Path) -> None:
    factory = _slurm_factory(condaenv="usas", mail_user="a@b.com", sbatch_args={"account": "myaccount"})

    job_name = "test"
    executor = factory.create(pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path), job_name=job_name)

    assert isinstance(executor, SlurmPipelineExecutor)
    assert executor.condaenv == "usas"
    assert executor.mail_user == "a@b.com"
    assert executor._sbatch_args == {"account": "myaccount"}
    assert executor.job_name == job_name


def test_create_slurm_backend_passes_depends_through(tmp_path: Path) -> None:
    factory = _slurm_factory()
    upstream = factory.create(pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path / "upstream"))

    downstream = factory.create(
        pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path / "downstream"), depends=upstream
    )

    assert isinstance(downstream, SlurmPipelineExecutor)
    assert downstream.depends is upstream


def test_create_unsupported_backend_raises_value_error(tmp_path: Path) -> None:
    factory = PipelineExecutorFactory(
        backend="invalid",  # ty: ignore[invalid-argument-type]
        randomize_start_duration=0,
        skip_completed=True,
    )

    with pytest.raises(ValueError, match="Unsupported executor backend"):
        factory.create(pipeline=[], tasks=1, workers=1, logging_dir=str(tmp_path))
