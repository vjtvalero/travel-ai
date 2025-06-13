import os
from openai import OpenAI

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "gpt-4.1-nano"

    user_memory = {}
    user_goal = input("Where are you going next? ")

    def finish(reason: str):
        print(f"✅ Agent finished: {reason}")

    def insert_memory(memory_name: str, content: str):
        user_memory[memory_name] = content
        print(f"📝 Memory '{memory_name}' inserted with content: {content}")

    def update_memory(memory_name: str, content: str):
        if memory_name in user_memory:
            user_memory[memory_name] = content
            print(f"📝 Memory '{memory_name}' updated with content: {content}")
        else:
            print(f"❗ Memory '{memory_name}' not found to update.")

    def fetch_memory(memory_name: str):
        if memory_name in user_memory:
            content = user_memory[memory_name]
            print(f"📖 Memory '{memory_name}' fetched with content: {content}")
            return content
        else:
            print(f"❗ Memory '{memory_name}' not found.")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "insert_memory",
                "description": "Insert a new memory item for the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_name": {"type": "string", "description": "Name of the memory item"},
                        "content": {"type": "string", "description": "Content of the memory item"},
                    },
                    "required": ["memory_name", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_memory",
                "description": "Update user memory with new information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_name": {"type": "string", "description": "Name of the memory item"},
                        "content": {"type": "string", "description": "Content of the memory item"},
                    },
                    "required": ["memory_name", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_memory",
                "description": "Fetch a memory item from the user's memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_name": {"type": "string", "description": "Name of the memory item"},
                    },
                    "required": ["memory_name"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": f'''
        You are a travel planning agent.
        Your goal is to assist the user in planning their next trip based on their preferences and requirements.
        You can use the following tools to help the user:
        - `insert_memory`: Insert a new memory item for the user.
        - `update_memory`: Update user memory with new information.
        - `fetch_memory`: Fetch a memory item from the user's memory.
        '''},
        {"role": "user", "content": user_goal}
    ]

    # Agent loop
    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        reply = response.choices[0].message
        messages.append(reply)

        if reply.tool_calls:
            for tool_call in reply.tool_calls:
                func_map = {
                    "insert_memory": (insert_memory, "Memory saved."),
                    "update_memory": (update_memory, "Memory updated."),
                    "fetch_memory": (fetch_memory, "Memory fetched."),
                }
                if tool_call.function.name in func_map:
                    func, msg = func_map[tool_call.function.name]
                    args = eval(tool_call.function.arguments)
                    func(**args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": msg
                    })
                elif tool_call.function.name == "finish":
                    args = eval(tool_call.function.arguments)
                    finish(**args)
                    exit(0)
        else:
            print("🤖 Agent message:", reply.content)

if __name__ == "__main__":
    main()