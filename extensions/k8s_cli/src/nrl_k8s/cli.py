# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI for managing nemo-rl workloads on Kubernetes with KAI scheduler."""

import click
from rich.console import Console
from rich.table import Table

from nrl_k8s.k8s_client import get_gpu_occupancy, get_queues, submit_gang_rayjob

console = Console()


@click.group()
def main():
    """nrl-k8s: Manage nemo-rl workloads on Kubernetes with KAI scheduler."""


@main.command()
def fairshare():
    """Show KAI scheduler queue fairshare configuration."""
    queues = get_queues()
    table = Table(title="KAI Scheduler Queues (Fairshare)")
    table.add_column("Queue", style="cyan")
    table.add_column("Parent", style="dim")
    table.add_column("Priority", justify="right")
    table.add_column("GPU Quota", justify="right")
    table.add_column("GPU Limit", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Preempt Min", style="dim")
    table.add_column("Reclaim Min", style="dim")

    for q in queues:
        table.add_row(
            q["name"],
            q["parent"] or "-",
            str(q["priority"]) if q["priority"] else "-",
            str(q["gpu_quota"]),
            str(q["gpu_limit"]),
            str(q["gpu_weight"]),
            q["preempt_min_runtime"] or "-",
            q["reclaim_min_runtime"] or "-",
        )

    console.print(table)


@main.command()
def occupancy():
    """Show current GPU occupancy per node and per queue."""
    data = get_gpu_occupancy()

    # Node table.
    node_table = Table(title="GPU Occupancy by Node")
    node_table.add_column("Node", style="cyan")
    node_table.add_column("Allocatable", justify="right")
    node_table.add_column("Allocated", justify="right")
    node_table.add_column("Free", justify="right", style="green")

    for n in data["nodes"]:
        if n["allocatable"] > 0:
            node_table.add_row(
                n["name"],
                str(n["allocatable"]),
                str(n["allocated"]),
                str(n["allocatable"] - n["allocated"]),
            )

    node_table.add_section()
    node_table.add_row(
        "TOTAL",
        str(data["total_allocatable"]),
        str(data["total_allocated"]),
        str(data["total_allocatable"] - data["total_allocated"]),
        style="bold",
    )
    console.print(node_table)

    # Queue table.
    if data["queues"]:
        queue_table = Table(title="GPU Occupancy by Queue")
        queue_table.add_column("Queue", style="cyan")
        queue_table.add_column("Allocated GPUs", justify="right")
        for q in data["queues"]:
            queue_table.add_row(q["name"], str(q["allocated_gpus"]))
        console.print(queue_table)
    else:
        console.print("[dim]No GPU workloads running.[/dim]")


@main.command()
@click.argument("name")
@click.option("--queue", required=True, help="KAI scheduler queue name")
@click.option("--image", required=True, help="Container image")
@click.option("--entrypoint", required=True, help="Entrypoint command")
@click.option("--num-gpus", required=True, type=int, help="Total GPUs requested")
@click.option("--gpus-per-worker", default=1, type=int, help="GPUs per worker pod")
@click.option("--namespace", default="default", help="Kubernetes namespace")
@click.option(
    "--segment-size",
    default=None,
    type=int,
    help="Topology segment size (nodes per rack). Creates PodGroup subgroups.",
)
def submit(
    name, queue, image, entrypoint, num_gpus, gpus_per_worker, namespace, segment_size
):
    """Submit a gang-scheduled RayJob."""
    console.print(
        f"Submitting RayJob [cyan]{name}[/cyan] to queue [yellow]{queue}[/yellow]"
    )
    console.print(
        f"  GPUs: {num_gpus} ({num_gpus // gpus_per_worker} workers × {gpus_per_worker} GPU/worker)"
    )
    if segment_size:
        import math

        num_segments = math.ceil(num_gpus // gpus_per_worker / segment_size)
        console.print(
            f"  Segments: {num_segments} × {segment_size} workers (topology-constrained per rack)"
        )

    result_name = submit_gang_rayjob(
        name=name,
        queue=queue,
        image=image,
        entrypoint=entrypoint,
        num_gpus=num_gpus,
        gpus_per_worker=gpus_per_worker,
        namespace=namespace,
        segment_size=segment_size,
    )
    console.print(f"[green]Created RayJob: {result_name}[/green]")
    console.print(f"Watch: kubectl get rayjob {result_name} -w")


if __name__ == "__main__":
    main()
