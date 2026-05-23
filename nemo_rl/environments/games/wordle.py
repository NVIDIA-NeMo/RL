# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import random
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

# Common 5-letter words for the word list
WORD_LIST = [
    "about", "above", "abuse", "actor", "acute", "admit", "adopt", "adult",
    "agent", "agree", "alarm", "album", "alert", "alien", "align", "alike",
    "alive", "alley", "allow", "alone", "along", "alter", "angel", "anger",
    "angle", "angry", "apart", "apple", "arena", "argue", "arise", "armor",
    "array", "aside", "asset", "avoid", "award", "aware", "basic", "beach",
    "begin", "being", "below", "bench", "berry", "birth", "black", "blade",
    "blame", "blank", "blast", "blaze", "bleed", "blend", "blind", "block",
    "blood", "bloom", "blown", "board", "bonus", "boost", "bound", "brain",
    "brand", "brave", "bread", "break", "breed", "brick", "brief", "bring",
    "broad", "broke", "brown", "brush", "build", "bunch", "burst", "buyer",
    "cabin", "cable", "candy", "carry", "catch", "cause", "chain", "chair",
    "charm", "chase", "cheap", "check", "cheek", "chest", "chief", "child",
    "china", "chose", "civil", "claim", "class", "clean", "clear", "climb",
    "cling", "clock", "close", "cloud", "coach", "coast", "count", "court",
    "cover", "crack", "craft", "crash", "crazy", "cream", "crime", "cross",
    "crowd", "cruel", "curve", "cycle", "daily", "dance", "death", "debug",
    "decay", "delay", "depth", "dirty", "doubt", "dough", "draft", "drain",
    "drama", "drank", "dream", "dress", "dried", "drift", "drink", "drive",
    "eager", "early", "earth", "eight", "elect", "elite", "empty", "enemy",
    "enjoy", "enter", "equal", "error", "event", "every", "exact", "exist",
    "extra", "faint", "faith", "fault", "feast", "fence", "fewer", "fiber",
    "field", "fifth", "fifty", "fight", "final", "first", "fixed", "flame",
    "flash", "fleet", "flesh", "float", "flood", "floor", "flour", "fluid",
    "focus", "force", "forge", "forth", "found", "frame", "frank", "fresh",
    "front", "frost", "fruit", "fully", "giant", "given", "glass", "globe",
    "glory", "going", "grace", "grade", "grain", "grand", "grant", "graph",
    "grasp", "grass", "grave", "great", "green", "greet", "grief", "grill",
    "grind", "gross", "group", "grove", "grown", "guard", "guess", "guide",
    "guilt", "habit", "happy", "harsh", "heart", "heavy", "hence", "horse",
    "hotel", "house", "human", "humor", "ideal", "image", "imply", "index",
    "inner", "input", "irony", "issue", "ivory", "jewel", "joint", "judge",
    "juice", "knife", "knock", "known", "label", "large", "laser", "later",
    "laugh", "layer", "learn", "lease", "least", "leave", "legal", "level",
    "light", "limit", "linen", "liver", "local", "loose", "lover", "lower",
    "lucky", "lunch", "magic", "major", "maker", "march", "match", "mayor",
    "media", "mercy", "metal", "meter", "might", "minor", "minus", "mixed",
    "model", "money", "month", "moral", "motor", "mount", "mouse", "mouth",
    "movie", "music", "naive", "nerve", "never", "night", "noble", "noise",
    "north", "noted", "novel", "nurse", "occur", "ocean", "offer", "often",
    "olive", "order", "other", "ought", "outer", "owner", "oxide", "paint",
    "panel", "panic", "paper", "patch", "pause", "peace", "peach", "penny",
    "phase", "phone", "photo", "piano", "piece", "pilot", "pitch", "pixel",
    "place", "plain", "plane", "plant", "plate", "plaza", "plead", "point",
    "polar", "pound", "power", "press", "price", "pride", "prime", "print",
    "prior", "prize", "proof", "proud", "prove", "psalm", "pupil", "queen",
    "quest", "queue", "quick", "quiet", "quote", "radar", "raise", "range",
    "rapid", "ratio", "reach", "ready", "realm", "rebel", "refer", "reign",
    "relax", "reply", "rider", "ridge", "rifle", "right", "rigid", "risky",
    "rival", "river", "robin", "robot", "rocky", "round", "route", "royal",
    "rugby", "rural", "saint", "salad", "sauce", "scale", "scene", "scope",
    "score", "sense", "serve", "seven", "shade", "shake", "shall", "shame",
    "shape", "share", "shark", "sharp", "sheep", "sheer", "sheet", "shelf",
    "shell", "shift", "shine", "shirt", "shock", "shoot", "shore", "short",
    "shout", "sight", "silly", "since", "sixth", "sixty", "skill", "skull",
    "slash", "slave", "sleep", "slice", "slide", "slope", "smart", "smell",
    "smile", "smoke", "snake", "solar", "solid", "solve", "sorry", "sound",
    "south", "space", "spare", "speak", "speed", "spend", "spill", "spine",
    "split", "sport", "spray", "squad", "stack", "staff", "stage", "stain",
    "stake", "stale", "stand", "stare", "start", "state", "steal", "steam",
    "steel", "steep", "steer", "stern", "stick", "stiff", "still", "stock",
    "stone", "stood", "store", "storm", "story", "stove", "strip", "stuck",
    "study", "stuff", "style", "sugar", "suite", "super", "surge", "swamp",
    "swear", "sweep", "sweet", "swift", "swing", "sword", "taste", "teach",
    "tempo", "thank", "theme", "thick", "thing", "think", "third", "those",
    "three", "threw", "throw", "thumb", "tiger", "tight", "timer", "tired",
    "title", "today", "token", "topic", "total", "touch", "tough", "towel",
    "tower", "trace", "track", "trade", "trail", "train", "trait", "trash",
    "treat", "trend", "trial", "tribe", "trick", "tried", "truck", "truly",
    "trump", "trunk", "trust", "truth", "tumor", "twice", "twist", "ultra",
    "under", "union", "unite", "unity", "until", "upper", "upset", "urban",
    "usage", "usual", "valid", "value", "vapor", "verse", "video", "vigor",
    "virus", "visit", "vital", "vivid", "vocal", "voice", "voter", "waste",
    "watch", "water", "weigh", "weird", "wheel", "where", "which", "while",
    "white", "whole", "whose", "width", "woman", "world", "worry", "worse",
    "worst", "worth", "would", "wound", "wrist", "write", "wrong", "wrote",
    "yield", "young", "youth",
]


class WordleMetadata(TypedDict):
    target_word: str
    guesses_made: int
    max_guesses: int
    feedback_history: list[str]


class WordleGameLogic:
    @staticmethod
    def generate(word_list: list[str] | None = None) -> dict[str, Any]:
        """Generate a new Wordle game by picking a random target word."""
        words = word_list if word_list else WORD_LIST
        target = random.choice(words)
        return {"target_word": target}

    @staticmethod
    def step(
        guess: str, game_state: dict[str, Any]
    ) -> tuple[str, float, bool, list[str]]:
        """Compare guess to target word, return feedback.

        Returns:
            Tuple of (feedback_string, reward, is_terminated, feedback_per_letter)
            feedback_per_letter: list of "G", "Y", or "X" for each letter
        """
        target = game_state["target_word"]
        guess = guess.lower().strip()

        if len(guess) != 5 or not guess.isalpha():
            return "Invalid guess. Must be exactly 5 letters.", 0.0, False, []

        # Compute feedback
        feedback = ["X"] * 5
        target_chars = list(target)

        # First pass: mark greens
        for i in range(5):
            if guess[i] == target[i]:
                feedback[i] = "G"
                target_chars[i] = None  # consumed

        # Second pass: mark yellows
        for i in range(5):
            if feedback[i] == "G":
                continue
            if guess[i] in target_chars:
                feedback[i] = "Y"
                target_chars[target_chars.index(guess[i])] = None  # consumed

        is_correct = all(f == "G" for f in feedback)
        reward = 1.0 if is_correct else 0.0

        feedback_str = " ".join(
            f"{guess[i].upper()}({feedback[i]})" for i in range(5)
        )
        return feedback_str, reward, is_correct, feedback

    @staticmethod
    def render(feedback_history: list[str]) -> str:
        """Render the feedback history grid."""
        if not feedback_history:
            return "No guesses yet."
        lines = []
        for i, entry in enumerate(feedback_history, 1):
            lines.append(f"  Guess {i}: {entry}")
        return "\n".join(lines)


class WordleRunner:
    def __init__(self):
        pass

    def _parse_guess(self, text: str) -> Optional[str]:
        """Parse guess from <guess>WORD</guess> tags."""
        text_lower = text.lower()
        start_idx = text_lower.rfind("<guess>")
        if start_idx != -1:
            end_idx = text_lower.find("</guess>", start_idx + len("<guess>"))
            if end_idx != -1:
                guess = text[start_idx + len("<guess>"):end_idx].strip()
                return guess
        return None

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: WordleMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[WordleMetadata],
        Optional[list[str]],
    ]:
        """Process a single turn for the Wordle task."""
        target_word = metadata["target_word"]
        guesses_made = metadata["guesses_made"]
        max_guesses = metadata["max_guesses"]
        feedback_history = list(metadata["feedback_history"])

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = ["</guess>"]
        next_metadata = metadata.copy()
        next_metadata["feedback_history"] = feedback_history

        # Check if max guesses reached
        if guesses_made >= max_guesses:
            is_terminated = True
            observation_content = (
                f"<environment>Maximum guesses ({max_guesses}) reached. "
                f"The word was: {target_word.upper()}</environment>"
            )
            return (
                {"role": "environment", "content": observation_content + "\n"},
                0.0,
                is_terminated,
                None,
                None,
                None,
            )

        # Get last assistant message and parse guess
        last_assistant_content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_content = message_log[-1]["content"].strip()

        parsed_guess = self._parse_guess(last_assistant_content)

        if parsed_guess is None:
            observation_content = (
                "<environment>Invalid format. Please wrap your guess in "
                "<guess>WORD</guess> tags.</environment>"
            )
            next_metadata = None
        else:
            game_state = {"target_word": target_word}
            feedback_str, reward, is_correct, feedback = WordleGameLogic.step(
                parsed_guess, game_state
            )

            if not feedback:
                # Invalid guess (wrong length / not alpha)
                observation_content = f"<environment>{feedback_str}</environment>"
                next_metadata = None
            else:
                turn_reward = reward
                is_terminated = is_correct
                feedback_history.append(feedback_str)
                next_metadata["guesses_made"] = guesses_made + 1
                next_metadata["feedback_history"] = feedback_history

                rendered = WordleGameLogic.render(feedback_history)
                remaining = max_guesses - (guesses_made + 1)

                if is_correct:
                    observation_content = (
                        f"<environment>\n{rendered}\n\n"
                        f"Correct! The word was {target_word.upper()}! "
                        f"Solved in {guesses_made + 1} guess(es).</environment>"
                    )
                    next_metadata = None
                elif remaining <= 0:
                    is_terminated = True
                    observation_content = (
                        f"<environment>\n{rendered}\n\n"
                        f"No guesses remaining. The word was: "
                        f"{target_word.upper()}</environment>"
                    )
                    next_metadata = None
                else:
                    observation_content = (
                        f"<environment>\n{rendered}\n\n"
                        f"{remaining} guess(es) remaining.\n"
                        f"G=correct position, Y=wrong position, X=not in word"
                        f"</environment>"
                    )

        next_answers = None
        return (
            {"role": "environment", "content": observation_content + "\n"},
            turn_reward,
            is_terminated,
            next_stop_strings,
            next_metadata,
            next_answers,
        )


@ray.remote  # pragma: no cover
class WordleEnv(EnvironmentInterface[WordleMetadata]):
    """Wordle environment (Ray Actor)."""

    def __init__(self, cfg: Optional[dict] = None):
        self.config = cfg or {}
        self.runner = WordleRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[WordleMetadata],
    ) -> EnvironmentReturn[WordleMetadata]:
        """Process a batch of Wordle interactions."""
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata)
        ]

        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []
        all_answers = []

        for obs, rew, term, stops, meta, answ in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)
            all_answers.append(answ)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=all_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        return batch, {"wordle_success_rate": success_rate}
