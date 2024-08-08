import json
import time
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="sk-proj-ALJj6LIQqZVp1wbQFKwFT3BlbkFJYwJw26yTw65BSTh0GFKs")

log_path = "./log/"
log_file = ""


def retry_until_expected(run, thread_id, expect):
    while True:
        run = client.beta.threads.runs.poll(thread_id=thread_id, run_id=run.id)
        if run.status == expect:
            break
        else:
            print(f"Got status {run.status}, try again in 60s...")
            time.sleep(60)


def get_last_text_message(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for message in messages.data:
        for content in message.content:
            if content.type == "text":
                return content.text.value


def log_to_file(type, message):
    if type == "input":
        message = (
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            + "\n"
        ) + message
    elif type == "output":
        message = (
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            + "\n"
        ) + message
    os.makedirs(log_path, exist_ok=True)
    with open(log_path + log_file, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def compare(index1_path, index2_path):
    with open("./data/prompt/0-instruction.md", "r", encoding="utf-8") as file:
        instruction = file.read()
    assistant = client.beta.assistants.create(
        name="Data Analysis Assistant",
        instructions=instruction,
        model="gpt-4o",
        tools=[
            {"type": "code_interpreter"},
            {
                "type": "function",
                "function": {
                    "name": "submit_better_alpha_index",
                    "description": "Submit the selected better alpha's index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "string", "description": "The index"}
                        },
                        "required": ["index"],
                    },
                },
            },
        ],
    )

    # Create a new thread
    thread = client.beta.threads.create()

    # first message
    with open("./data/prompt/1-preamble.md", "r", encoding="utf-8") as file:
        content = file.read()
    log_to_file("input", content)
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        # Code Interpeter has access to matplotlib, letting you plot values
        # in addition to answering questions
        content=content,
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    retry_until_expected(run, thread.id, "completed")
    log_to_file("output", get_last_text_message(thread.id))

    # second message - 上证50
    with open("./data/prompt/2-上证50.md", "r", encoding="utf-8") as file:
        content = file.read()
    log_to_file("input", content)
    file_id = client.files.create(
        file=open("./data/上证50.xlsx", "rb"), purpose="assistants"
    ).id
    log_to_file("input", "./data/上证50.xlsx")
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        # Code Interpeter has access to matplotlib, letting you plot values
        # in addition to answering questions
        content=content,
        attachments=[{"file_id": file_id, "tools": [{"type": "code_interpreter"}]}],
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    retry_until_expected(run, thread.id, "completed")
    log_to_file("output", get_last_text_message(thread.id))

    # third message - index 1
    with open("./data/prompt/3-index1.md", "r", encoding="utf-8") as file:
        content = file.read()
    log_to_file("input", content)
    file1_id = client.files.create(
        file=open(index1_path, "rb"), purpose="assistants"
    ).id
    log_to_file("input", index1_path)
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        # Code Interpeter has access to matplotlib, letting you plot values
        # in addition to answering questions
        content=content,
        attachments=[{"file_id": file1_id, "tools": [{"type": "code_interpreter"}]}],
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    retry_until_expected(run, thread.id, "completed")
    log_to_file("output", get_last_text_message(thread.id))

    # fourth message - index 2
    with open("./data/prompt/4-index2.md", "r", encoding="utf-8") as file:
        content = file.read()
    log_to_file("input", content)
    file2_id = client.files.create(
        file=open(index2_path, "rb"), purpose="assistants"
    ).id
    log_to_file("input", index2_path)
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        # Code Interpeter has access to matplotlib, letting you plot values
        # in addition to answering questions
        content=content,
        attachments=[{"file_id": file2_id, "tools": [{"type": "code_interpreter"}]}],
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    retry_until_expected(run, thread.id, "requires_action")
    log_to_file("output", get_last_text_message(thread.id))
    # Define the list to store tool outputs
    tool_outputs = []

    # Loop through each tool in the required action section
    for tool in run.required_action.submit_tool_outputs.tool_calls:
        if tool.function.name == "submit_better_alpha_index":
            tool_outputs.append({"tool_call_id": tool.id, "output": "success"})
    try:
        run = client.beta.threads.runs.submit_tool_outputs_and_poll(
            thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
        )
    except Exception as e:
        print("Failed to submit tool outputs:", e)
    log_to_file("output", get_last_text_message(thread.id))
    index = json.loads(tool.to_dict()["function"]["arguments"])["index"]
    log_to_file("output", f"The selected better alpha's index is: {index}")
    return index


# list files in ./data/alpha-result/
import os

files = os.listdir("./data/alpha-result/")
files = [f for f in files if f.endswith(".xlsx")]

best_file = files[0]
best_file_index = 1
round = 1
for i, file in enumerate(tqdm(files[1:])):
    index = i + 2
    log_file = f"round-{round}-{best_file_index}-{index}.log"
    best_index = compare(
        f"./data/alpha-result/{best_file}", f"./data/alpha-result/{file}"
    )
    if best_index == "2":
        best_file = file
        best_file_index = index
    round += 1

print(f"The best alpha is: {best_file}")
