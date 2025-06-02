import os
import subprocess

VISIBLE_DEVICES = [7]

datasets = [
    "nfcorpus",
    "scifact",
    "arguana",
    "scidocs",
    "fiqa",
    "trec-covid",
    "webis-touche2020",
    "quora",
    "nq",
    "dbpedia-entity",
    "hotpotqa",
    "msmarco",
]

if not VISIBLE_DEVICES:
    error = (
        "The VISIBLE_DEVICES list cannot be empty. Please specify at least one GPU ID."
    )
    raise ValueError(error)

NUM_GPUS = len(VISIBLE_DEVICES)

processes = []

print(
    f"🚀 Launching evaluations for {len(datasets)} datasets using a pool of {NUM_GPUS} GPUs (IDs: {VISIBLE_DEVICES})."  # noqa: E501
)
print("A new job will start as soon as a slot in the GPU pool is available.\n")

for i, dataset in enumerate(datasets):
    logical_gpu_index = i % NUM_GPUS
    physical_gpu_id = VISIBLE_DEVICES[logical_gpu_index]

    if i >= NUM_GPUS:
        p_to_wait_on, d_to_wait_on = processes[i - NUM_GPUS]
        print(
            f"  -> GPU pool is full. Waiting for job '{d_to_wait_on}' to free up GPU {physical_gpu_id}..."  # noqa: E501
        )
        p_to_wait_on.wait()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    command = ["python", "benchmark/benchmark.py", "--dataset", dataset]

    print(f"  -> Launching '{dataset}' on GPU {physical_gpu_id}...")
    process = subprocess.Popen(command, env=env)
    processes.append((process, dataset))

print(
    "\n... All jobs have been launched. Waiting for the final running jobs to complete ..."
)

for process, _ in processes:
    process.wait()

print("\n--- Final Job Status ---")
for process, dataset in processes:
    if process.returncode == 0:
        print(f"✅ Successfully completed evaluation for '{dataset}'")
    else:
        print(
            f"❌ Evaluation for '{dataset}' FAILED with return code {process.returncode}"
        )

print("\n🎉 All evaluations are complete.")
