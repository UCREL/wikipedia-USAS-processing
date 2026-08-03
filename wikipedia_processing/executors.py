from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, cast

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.base import PipelineStep


class ExecutorBackend(str, Enum):
    """Which DataTrove executor backend to run pipeline stages with.

    Attributes:
        local: Run each stage as local multiprocessing workers, via
            `datatrove.executor.LocalPipelineExecutor`.
        slurm: Submit each stage as a Slurm job array, via
            `datatrove.executor.slurm.SlurmPipelineExecutor`.
    """

    local = "local"
    slurm = "slurm"


@dataclass
class SlurmExecutorSettings:
    """Slurm-specific settings shared by every stage of a pipeline run.

    Attributes:
        partition: Slurm partition to submit jobs to.
        time: Slurm job time limit, e.g. "2:00:00".
        cpus_per_task: Number of CPUs to request per Slurm task.
        mem_per_cpu_gb: Memory in GB to request per CPU.
        qos: Slurm QOS to submit jobs under.
        venv_path: Path to a virtualenv to activate in each Slurm job.
            Mutually exclusive with `condaenv`.
        condaenv: Name of a conda environment to activate in each Slurm
            job. Mutually exclusive with `venv_path`.
        mail_user: Email address for Slurm job notifications.
        mail_type: Slurm mail notification type(s), e.g. "ALL", "FAIL".
        sbatch_args: Additional raw sbatch arguments to pass through,
            e.g. `{"account": "myaccount"}`.
    """

    partition: str
    time: str
    cpus_per_task: int = 1
    mem_per_cpu_gb: int = 2
    qos: str = "normal"
    venv_path: Path | None = None
    condaenv: str | None = None
    mail_user: str | None = None
    mail_type: str = "ALL"
    sbatch_args: dict[str, str] | None = None


class PipelineExecutorFactory:
    """Builds DataTrove pipeline executors for a single backend, local or Slurm.

    Each pipeline stage shares the same backend and the same
    `randomize_start_duration` / `skip_completed` settings, but differs in
    its pipeline steps, task/worker counts, logging directory and
    dependencies. `create` captures the shared settings once and takes the
    per-stage arguments at each call site.

    Examples:
        >>> factory = PipelineExecutorFactory(
        ...     backend=ExecutorBackend.local,
        ...     randomize_start_duration=0,
        ...     skip_completed=False,
        ... )
        >>> executor = factory.create(pipeline=[], tasks=1, workers=1, logging_dir="/tmp")
        >>> isinstance(executor, LocalPipelineExecutor)
        True
    """

    def __init__(
        self,
        backend: ExecutorBackend,
        randomize_start_duration: int,
        skip_completed: bool,
        slurm_settings: SlurmExecutorSettings | None = None,
    ) -> None:
        """Initialize the factory for a given backend.

        Args:
            backend: Which executor backend every stage should be built for.
            randomize_start_duration: Maximum number of seconds to delay the
                start of each task, passed through to every stage's executor.
            skip_completed: Whether stages should skip already-completed
                tasks, passed through to every stage's executor.
            slurm_settings: Slurm-specific settings, required when
                `backend` is `ExecutorBackend.slurm` and unused otherwise.

        Raises:
            ValueError: If `backend` is `ExecutorBackend.slurm` and
                `slurm_settings` is None.
        """
        if backend is ExecutorBackend.slurm and slurm_settings is None:
            raise ValueError("slurm_settings is required when backend is ExecutorBackend.slurm")
        self.backend = backend
        self.randomize_start_duration = randomize_start_duration
        self.skip_completed = skip_completed
        self.slurm_settings = slurm_settings

    def create(
        self,
        pipeline: list[PipelineStep | Callable],
        tasks: int,
        workers: int,
        logging_dir: DataFolderLike,
        depends: PipelineExecutor | None = None,
        job_name: str = "data_processing",
        tasks_per_job: int = 1,
    ) -> PipelineExecutor:
        """Build a single pipeline stage's executor for this factory's backend.

        Args:
            pipeline: The ordered list of pipeline steps (or callables) to run.
            tasks: The number of tasks to split the stage's work into.
            workers: The number of tasks to run concurrently.
            logging_dir: Directory to write the stage's executor logs to.
            depends: The executor of the stage this one depends on, if any.
                Must have been built by a factory with the same backend.
            job_name: Human-readable name for the stage. Only used by the
                Slurm backend, to identify the stage's jobs in `squeue`.
            tasks_per_job: How many tasks each submitted Slurm array element
                runs, sequentially. Reduces the number of Slurm array
                elements submitted for this stage from `tasks` to
                `ceil(tasks / tasks_per_job)`. Only used by the Slurm
                backend; ignored (no equivalent) on the local backend.

        Returns:
            A `LocalPipelineExecutor` or `SlurmPipelineExecutor`, depending
            on this factory's backend.
        """
        match self.backend:
            case ExecutorBackend.local:
                # datatrove types `depends` as plain `LocalPipelineExecutor` even though
                # its actual default is None, so the `LocalPipelineExecutor | None` value
                # here is cast to satisfy the (imprecise) upstream annotation.
                return LocalPipelineExecutor(
                    pipeline=pipeline,
                    tasks=tasks,
                    workers=workers,
                    logging_dir=logging_dir,
                    depends=cast(LocalPipelineExecutor, depends),
                    skip_completed=self.skip_completed,
                    randomize_start_duration=self.randomize_start_duration,
                )
            case ExecutorBackend.slurm:
                assert self.slurm_settings is not None
                settings = self.slurm_settings
                venv_path = str(settings.venv_path.resolve()) if settings.venv_path is not None else None
                # datatrove types venv_path/condaenv/mail_user as plain `str` even though
                # their actual default is None, so the `str | None` values here are cast
                # to satisfy the (imprecise) upstream annotation.
                return SlurmPipelineExecutor(
                    pipeline=pipeline,
                    tasks=tasks,
                    workers=workers,
                    logging_dir=logging_dir,
                    depends=cast(SlurmPipelineExecutor | None, depends),
                    skip_completed=self.skip_completed,
                    randomize_start_duration=self.randomize_start_duration,
                    job_name=job_name,
                    partition=settings.partition,
                    time=settings.time,
                    cpus_per_task=settings.cpus_per_task,
                    mem_per_cpu_gb=settings.mem_per_cpu_gb,
                    qos=settings.qos,
                    venv_path=cast(str, venv_path),
                    condaenv=cast(str, settings.condaenv),
                    mail_user=cast(str, settings.mail_user),
                    mail_type=settings.mail_type,
                    sbatch_args=settings.sbatch_args,
                    tasks_per_job=tasks_per_job,
                )
            case _:
                raise ValueError(f"Unsupported executor backend: {self.backend!r}")
