# Turn a Workplace Assistant example row into one forced calendar_create_event
# call that writes the recovery sentinel event. Shared by the single-controller
# Gym recovery functional tests (jq -L tests/functional, include this module).
#
# $instruction is appended to the prompt. $duration is the ground-truth
# duration; the prompt always asks for 30, so any other value scores 0.
def workplace_calendar_row($event; $instruction; $duration):
  .task_source = "workplace_assistant_checkpoint_test_agent"
  | .responses_create_params.input = [{
      "role": "user",
      "content": ("Call calendar_create_event exactly once with event_name " + $event
        + ", participant_email checkpoint-recovery@example.com, event_start "
        + "2025-01-15 10:00:00, and duration 30." + $instruction)
    }]
  | .responses_create_params.tools = [
      .responses_create_params.tools[] | select(.name == "calendar_create_event")
    ]
  | .responses_create_params.tool_choice = {"type": "function", "name": "calendar_create_event"}
  | .responses_create_params.parallel_tool_calls = false
  | .ground_truth = [{
      "name": "calendar_create_event",
      "arguments": ({
        "event_name": $event,
        "participant_email": "checkpoint-recovery@example.com",
        "event_start": "2025-01-15 10:00:00",
        "duration": $duration
      } | tojson)
    }]
  | .category = "workplace_assistant_calendar";
