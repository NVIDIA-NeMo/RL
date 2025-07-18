import asyncio

from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# Define the agents
def create_agent(instruction: str | None = None, name: str = "simulated_user", model: str = 'gemini-2.0-flash') -> Agent:
    return Agent(
        model=model,
        name=name,
        description="Agent",
        instruction=instruction or "You are a helpful assistant that help people answer questions.",
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ]
        ),
    )

def get_session_from_runner(runner: Runner, user_id: str):
    app_session_map = runner.session_service.sessions
    assert len(app_session_map) == 1, "Expected exactly one app in session_service"
    user_sessions_map = next(iter(app_session_map.values()))
    sessions = user_sessions_map[user_id]
    assert len(sessions) == 1, "Expected exactly one user in app session"
    return next(iter(sessions.values()))

def get_agent_instruction_from_runner(runner: Runner):
    return runner.agent.instruction

def extract_conversation_history(runner: Runner, user_id: str, silence: bool = True):
    session = get_session_from_runner(runner, user_id)
    instruction = get_agent_instruction_from_runner(runner)
    convo = [{"role": "instruction", "content":instruction}]
    for event in session.events:
        if event.content.parts and event.content.parts[0].text:
            convo.append({"role": event.author, "content": event.content.parts[0].text})
            if not silence:
                print(f"[{convo[-1]['role']}]: {convo[-1]['content']}")
    return session.id, convo


async def run_prompt_async(runner: Runner, user_id: str, new_message: str, silence: bool = True):
    content = types.Content(role='user', parts=[types.Part.from_text(text=new_message)])
    if not silence:
        print('** User says:', new_message)

    session = get_session_from_runner(runner, user_id)

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.content.parts and event.content.parts[0].text:
            if not silence:
                print(f'** {event.author} says: {event.content.parts[0].text}')
            return event.content.parts[0].text.strip()
        
    return "<no response>" 

async def setup_runner_async(agent: Agent, app_name: str, user_id: str):
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=InMemorySessionService()
    )
    await runner.session_service.create_session(app_name=app_name, user_id=user_id)
    return runner


async def main():

    sample_id_1 = "sample_1"
    sample_id_2 = "sample_2"

    # Set up simulated user runner
    simulated_user_app_name = "su_app"
    simulated_user_runner = Runner(
        agent=create_agent(name="simulated_user"),
        app_name=simulated_user_app_name,
        session_service=InMemorySessionService()
    )

    await simulated_user_runner.session_service.create_session(app_name=simulated_user_app_name, user_id=sample_id_1)
    await simulated_user_runner.session_service.create_session(app_name=simulated_user_app_name, user_id=sample_id_2)

    # setup grader runner
    grader_app_name = "grader_app"
    grader_instruction = "You are a helpful agent that can grade the correctness and coherent of a conversation. Please only give an integer as the score."
    grader_runner = await setup_runner_async(agent=create_agent(name="grader", instruction=grader_instruction), app_name=grader_app_name, user_id=sample_id_1)

    # Simulated user interactions
    await run_prompt_async(simulated_user_runner, sample_id_1, 'what is 2*3+5?', silence=False)
    await run_prompt_async(simulated_user_runner, sample_id_2, 'what is 2*3-5?')
    await run_prompt_async(simulated_user_runner, sample_id_1, 'Now add another 10.')
    await run_prompt_async(simulated_user_runner, sample_id_2, 'Now add another 100.')

    # Print conversation
    print("-" * 100)
    _, convo1 = extract_conversation_history(simulated_user_runner, sample_id_1, silence=False)
    print("-" * 100)
    _, convo2 = extract_conversation_history(simulated_user_runner, sample_id_2, silence=False)
    print("-" * 100)

    # Grade conversation
    await run_prompt_async(grader_runner, sample_id_1, f'Grade the above conversation and give a score between 0-10. \n\n{convo1}', silence=False)
    print("-" * 100)
    print("DONE!")


if __name__ == "__main__":
    asyncio.run(main())