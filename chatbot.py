import os
import instructor
from rich.console import Console
from rich.text import Text
from rich.live import Live
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema
from dotenv import load_dotenv


load_dotenv()

# Initialize console for pretty outputs
console = Console()

# Memory setup
memory = AgentMemory()

# Initialize memory with an initial message from the assistant
initial_message = BaseAgentOutputSchema(chat_message="Hello! How can I assist you today?")
memory.add_message("assistant", initial_message)

# Function to set up the client based on the chosen provider
def setup_client(provider):
    if provider == "openai":
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(openai.AsyncOpenAI(api_key=api_key))
        model = "gpt-4o-mini"

    elif provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = instructor.from_anthropic(anthropic.AsyncAnthropic(api_key=api_key))
        model = "claude-3-5-haiku-20241022"

    elif provider == "groq":
        import groq
        api_key = os.getenv("GROQ_API_KEY")
        client = instructor.from_groq(
            groq.AsyncGroq(api_key=api_key),
            mode=instructor.Mode.JSON
        )
        model = "mixtral-8x7b-32768"

    elif provider == "ollama":
        import openai
        client = instructor.from_openai(
            openai.AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            ),
            mode=instructor.Mode.JSON
        )
        model = "llama3"

    elif provider == "gemini":
        import openai
        api_key = os.getenv("GEMINI_API_KEY")
        client = instructor.from_openai(
            openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            mode=instructor.Mode.JSON
        )
        model = "gemini-2.0-flash-exp"
        
    elif provider == "openrouter":
        import openai
        api_key = os.getenv("OPENROUTER_API_KEY")
        client = instructor.from_openai(
            openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
        )
        model = "mistral/ministral-8b"

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client, model

# Prompt for provider choice
provider = console.input("Choose a provider (openai/anthropic/groq/ollama/gemini/openrouter): ").lower()

# Set up client and model
client, model = setup_client(provider)

# Create agent with chosen provider
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model=model,
        memory=memory,
        model_api_parameters={"max_tokens": 2048}
    )
)


# Display the initial message from the assistant
console.print(Text("Agent:", style="bold green"), end=" ")
console.print(Text(initial_message.chat_message, style="green"))

async def main():
    # Start an infinite loop to handle user inputs and agent responses
    while True:
        # Prompt the user for input with a styled prompt
        user_input = console.input("\n[bold blue]You:[/bold blue] ")
        # Check if the user wants to exit the chat
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("Exiting chat...")
            break

        # Process the user's input through the agent and get the streaming response
        input_schema = BaseAgentInputSchema(chat_message=user_input)
        console.print()  # Add newline before response

        # Use Live display to show streaming response
        with Live("", refresh_per_second=10, auto_refresh=True) as live:
            current_response = ""
            async for partial_response in agent.run_async(input_schema):
                if hasattr(partial_response, "chat_message") and partial_response.chat_message:
                    # Only update if we have new content
                    if partial_response.chat_message != current_response:
                        current_response = partial_response.chat_message
                        # Combine the label and response in the live display
                        display_text = Text.assemble(("Agent: ", "bold green"), (current_response, "green"))
                        live.update(display_text)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())