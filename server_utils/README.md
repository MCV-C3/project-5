# Running code on the GPU server

`manual.pdf` provides a detailed description of how to work with the server.

**Some guidelines:**

- Check the status of the GPUs with `nvitop`.
- `run_job.sh` is an example of how to run jobs using `slurm`.
- Use `job_mhigh` to queue jobs with max priority, 1 at a time.
- Use `job_mlow` to take profit of free resources if available.