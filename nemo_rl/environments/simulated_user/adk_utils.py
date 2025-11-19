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

import asyncio
import logging
import random

# Initialize logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)


# Define the agents
def create_agent(
    instruction: str | None = None,
    name: str = "simulated_user",
    model: str = "gemini-2.0-flash",
):
    from google.adk.agents import Agent
    from google.genai import types

    return Agent(
        model=model,
        name=name,
        description="Agent",
        instruction=instruction
        or "You are a helpful assistant that help people answer questions.",
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ]
        ),
    )


def get_session_from_runner(runner, user_id: str):
    app_session_map = runner.session_service.sessions
    assert len(app_session_map) == 1, "Expected exactly one app in session_service"
    user_sessions_map = next(iter(app_session_map.values()))
    sessions = user_sessions_map[user_id]
    assert len(sessions) == 1, "Expected exactly one user in app session"
    return next(iter(sessions.values()))


def get_agent_instruction_from_runner(runner):
    return runner.agent.instruction


def extract_conversation_history(runner, user_id: str, silence: bool = True):
    session = get_session_from_runner(runner, user_id)
    instruction = get_agent_instruction_from_runner(runner)
    convo = [{"role": "instruction", "content": instruction}]
    for event in session.events:
        if event.content.parts and event.content.parts[0].text:
            convo.append({"role": event.author, "content": event.content.parts[0].text})
            if not silence:
                logger.info(f"[{convo[-1]['role']}]: {convo[-1]['content']}")
    return session.id, convo


async def run_prompt_async(
    runner,
    user_id: str,
    new_message: str,
    silence: bool = True,
    max_retries: int = 3,
    initial_delay: float = 2,
) -> str:
    from google.genai import types
    from google.genai.errors import ServerError

    new_message = new_message.strip()
    content = types.Content(role="user", parts=[types.Part.from_text(text=new_message)])
    if not silence:
        logger.info(f"** [User]->|||{new_message}|||")

    session = get_session_from_runner(runner, user_id)

    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content,
            ):
                if event.content.parts and event.content.parts[0].text:
                    if not silence:
                        logger.info(
                            f"** [{event.author}]->|||{event.content.parts[0].text.strip()}|||"
                        )
                    return event.content.parts[0].text.strip()
                else:
                    return "<Empty response>"
        except ServerError as e:
            retries += 1
            delay_with_jitter = delay + (random.random() * 2 - 1) * (delay * 0.5)
            logger.error(
                f"Gemini API call (with message {new_message}) failed with ServerError {e} (attempt {retries}/{max_retries}). Retrying in {delay_with_jitter} seconds..."
            )
            await asyncio.sleep(delay_with_jitter)
            delay *= 2  # Exponential backoff
        except Exception as e:
            logger.error(
                f"Gemini API call (with message {new_message}) failed with an unexpected error: {e}."
            )
            return f"<No response due to unexpected error: {e}>"

    logger.error(
        f"Gemini API call (with message {new_message}) reached maximum retries ({max_retries}) without success."
    )
    return f"<No response due after {max_retries} retries>"


async def setup_runner_async(agent, app_name: str, user_id: str):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    runner = Runner(
        agent=agent, app_name=app_name, session_service=InMemorySessionService()
    )
    await runner.session_service.create_session(app_name=app_name, user_id=user_id)
    return runner


async def main():
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    sample_id_1 = "sample_1"
    sample_id_2 = "sample_2"

    # Set up simulated user runner
    simulated_user_app_name = "su_app"
    simulated_user_runner = Runner(
        agent=create_agent(name="simulated_user"),
        app_name=simulated_user_app_name,
        session_service=InMemorySessionService(),
    )

    await simulated_user_runner.session_service.create_session(
        app_name=simulated_user_app_name, user_id=sample_id_1
    )
    await simulated_user_runner.session_service.create_session(
        app_name=simulated_user_app_name, user_id=sample_id_2
    )

    # setup grader runner
    grader_app_name = "grader_app"
    grader_instruction = "You are a helpful agent that can grade the correctness and coherent of a conversation. Please only give an integer as the score."
    grader_runner = await setup_runner_async(
        agent=create_agent(name="grader", instruction=grader_instruction),
        app_name=grader_app_name,
        user_id=sample_id_1,
    )

    # Simulated user interactions
    await run_prompt_async(
        simulated_user_runner, sample_id_1, "what is 2*3+5?", silence=False
    )
    await run_prompt_async(simulated_user_runner, sample_id_2, "what is 2*3-5?")
    await run_prompt_async(simulated_user_runner, sample_id_1, "Now add another 10.")
    await run_prompt_async(simulated_user_runner, sample_id_2, "Now add another 100.")

    # Print conversation
    logger.info("-" * 100)
    _, convo1 = extract_conversation_history(
        simulated_user_runner, sample_id_1, silence=False
    )
    logger.info("-" * 100)
    _, convo2 = extract_conversation_history(
        simulated_user_runner, sample_id_2, silence=False
    )
    logger.info("-" * 100)

    # Grade conversation
    await run_prompt_async(
        grader_runner,
        sample_id_1,
        f"Grade the above conversation and give a score between 0-10. \n\n{convo1}",
        silence=False,
    )
    logger.info("-" * 100)
    logger.info("DONE!")


if __name__ == "__main__":
    asyncio.run(main())
