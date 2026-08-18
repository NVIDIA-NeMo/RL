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

"""Rollout tree: one TurnNode per turn, and the compaction hook."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional


@dataclass
class TurnNode:
    """One turn of a rollout.

    ``items`` holds that turn's Responses-API items in order: the model output,
    then its tool results, then any steering messages. The root node holds the
    prompt instead. A branch is a root-to-leaf path, and its content is the
    concatenation of the ``items`` along it.
    """

    turn_index: int
    items: list[dict[str, Any]] = field(default_factory=list)
    parent: Optional["TurnNode"] = None
    children: list["TurnNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, items: Optional[list[dict[str, Any]]]) -> None:
        if items:
            self.items.extend(items)

    def add_child(self, **metadata: Any) -> "TurnNode":
        child = TurnNode(turn_index=self.turn_index + 1, parent=self, metadata=metadata)
        self.children.append(child)
        return child

    def path_from_root(self) -> list["TurnNode"]:
        path: list["TurnNode"] = []
        node: Optional["TurnNode"] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _flatten(nodes: list[TurnNode]) -> list[dict[str, Any]]:
    return [item for node in nodes for item in node.items]


class BaseRolloutTree:
    """Owns the turn tree and decides what the model sees on the next turn.

    Two views, because they are not the same list:

    - ``get_active_branch()`` is the Responses-API ``input`` for the next model
      call: the prompt plus every turn along the active path.
    - ``get_active_outputs()`` is the ``response.output`` portion, which is what
      ``/verify`` and the NeMo-RL postprocessor read. It drops the root.

    Traversal is shared here; a subclass only decides what
    ``branch_and_compact_context`` does at the end of a turn: advance along the
    same branch, fork a new one, or compact before doing either.
    ``model_client`` is an async callable taking a Responses-API request body and
    returning the response JSON, so a compacting subclass can summarize context
    on its own without going back through the agent loop.
    """

    def __init__(
        self,
        prompt: list[dict[str, Any]],
        model_client: Optional[Callable[..., Any]] = None,
        cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        self.root = TurnNode(
            turn_index=0, items=list(prompt), metadata={"kind": "prompt"}
        )
        self.model_client = model_client
        self.cfg = cfg or {}
        # The root holds the prompt and nothing else, so the first turn gets its
        # own node and never mixes generated items into the prompt.
        self.active = self.root
        self.start_new_turn(kind="turn")

    def get_active_branch(self) -> list[dict[str, Any]]:
        """Items to send as ``input`` on the next model call."""
        return _flatten(self.active.path_from_root())

    def get_active_outputs(self) -> list[dict[str, Any]]:
        """Items to report as ``response.output`` for the active branch."""
        return self.outputs_for(self.active)

    def append(self, items: Optional[list[dict[str, Any]]]) -> None:
        """Add model output, tool results, or steering messages to the open turn."""
        self.active.append(items)

    def start_new_turn(
        self, parent: Optional[TurnNode] = None, **metadata: Any
    ) -> TurnNode:
        """Open a new turn under ``parent`` (the active node by default)."""
        self.active = (parent or self.active).add_child(**metadata)
        return self.active

    def outputs_for(self, node: TurnNode) -> list[dict[str, Any]]:
        """The ``response.output`` of the branch ending at ``node``."""
        return _flatten(node.path_from_root()[1:])

    def walk(self, node: Optional[TurnNode] = None) -> Iterator[TurnNode]:
        node = node or self.root
        yield node
        for child in node.children:
            yield from self.walk(child)

    def leaves(self) -> list[TurnNode]:
        return [node for node in self.walk() if node.is_leaf]

    def return_all_rollouts(self) -> list[list[dict[str, Any]]]:
        """One ``response.output`` list per branch, i.e. per trainable sequence."""
        return [self.outputs_for(leaf) for leaf in self.leaves()]

    def branch_and_compact_context(self) -> None:
        """End-of-turn hook: may compact, may fork, must open the next turn."""
        raise NotImplementedError
