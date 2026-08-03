from datatrove.pipeline.writers import HuggingFaceDatasetWriter


class ReusableHuggingFaceDatasetWriter(HuggingFaceDatasetWriter):
    """A `HuggingFaceDatasetWriter` safe to reuse across multiple ranks in one process.

    `SlurmPipelineExecutor`'s `tasks_per_job` option (see
    `wikipedia_processing.executors`) bundles several ranks into a single
    submitted Slurm array element, which runs them sequentially in the same
    Python process against the *same* pipeline step instances -- including
    the same writer instance (see `PipelineExecutor._run_for_rank` and
    `SlurmPipelineExecutor.run`).

    `HuggingFaceDatasetWriter.close` accumulates uploaded files onto
    `self.operations` and commits that list, but never clears it afterwards.
    When the same instance is reused for a later rank, its `close` call
    re-includes `CommitOperationAdd` objects from the earlier rank's
    already-successful commit. `huggingface_hub` rejects committing the same
    `CommitOperationAdd` twice, raising: "CommitOperationAdd ... has already
    being committed and cannot be reused." This subclass clears
    `self.operations` after each successful commit so the next rank (if any)
    starts from a clean slate, matching the intended one-commit-per-rank
    behavior.
    """

    def close(self, rank: int = 0) -> None:
        super().close(rank)
        self.operations = []
