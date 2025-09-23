import os

from datatrove.executor import SlurmPipelineExecutor


class C5SlurmExecutor(SlurmPipelineExecutor):
    """
    Custom Slurm Executor for C5 project.
    The only change is that we allow cpus_per_task and mem_per_cpu_gb to be None
    so that we don't pass them to sbatch if they are not set.
    That should give us sensible defaults from slurm side.
    """

    def get_sbatch_args(self, max_array: int = 1) -> dict:
        """
            Get a dictionary with all the sbatch directives we want to include
        Args:
            max_array: max array size

        Returns: a dictionary with all the sbatch directives

        """
        # this one we actually have to create as slurm will be writing here
        os.makedirs(self.slurm_logs_folder, exist_ok=True)
        slurm_logfile = os.path.join(self.slurm_logs_folder, "%A_%a.out")
        sbatch_args = {
            "partition": self.partition,
            "job-name": self.job_name,
            "time": self.time,
            "output": slurm_logfile,
            "error": slurm_logfile,
            "array": f"0-{max_array - 1}{f'%{self.workers}' if self.workers != -1 else ''}",
            **({"mail-type": self.mail_type, "mail-user": self.mail_user} if self.mail_user else {}),
            **self._sbatch_args,
        }
        if self.cpus_per_task:
            sbatch_args["cpus-per-task"] = self.cpus_per_task
        if self.mem_per_cpu_gb:
            sbatch_args["mem-per-cpu"] = f"{self.mem_per_cpu_gb}G"

        if self.requeue:
            sbatch_args["requeue"] = ""
        if self.qos:
            sbatch_args["qos"] = self.qos
        return sbatch_args
