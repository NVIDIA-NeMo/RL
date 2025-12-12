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

starting_user_prompt = (
    "I will play a game with you. I have a list of integers in mind and can NOT tell you. "
    "Your goal is to guess the count of UNIQUE numbers in my list. The only 2 things you can do is the following: "
    "You can either ask me 'what is number k?' to get the number at position k in my list, "
    "or answer 'there are m unique numbers' whenever you feel you want to make a guess. "
    "Please do not say anything else. You cannot ask me to provide the list of integers."
)


simulated_user_instruction = """
You are a simulated user in a game where the assistant must figure out how many unique numbers you have.
You have a list of numbers (which may contain duplicates) that you will not reveal to the assistant.
The assistant can ask you questions of the form "What is number k?" where k is a 1-based index into your list of numbers.
You should respond with the number at that index.
The assistant can also make a guess by saying "There are m unique numbers" where m is their guess for the count of unique numbers.
If the assistant makes a correct guess, you will reward it. If the guess is incorrect, you will penalize it.

Here is your list of numbers: {numbers}.
""".strip()

grader_instruction = """
Your are a strict grader to evaluate whether the assistant has properly guessed the count of unique numbers.
Here is your list of numbers: {numbers}.
You will see a conversation between the assistant and a simulated user who has this list of numbers.
You will need to evaluete in the end whether the assistant has made a correct guess of the count of unique numbers.
If the assistant made a correct guess, give it a score of 1. If the guess is incorrect, give it a score of 0.
If assistant made a correct guess but you feel the assistant has asked too many questions, please give a score between 0 and 1.
If the assistant never made a guess, give it a score of 0.
Please only output an integer score between 0 and 1, and nothing else.
""".strip()
