import os
import dotenv
import instructor
from instructor import Image
import openai
from rich.console import Console
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema

# Initialize console for pretty outputs
console = Console()

# Memory setup
memory = AgentMemory()

dotenv.load_dotenv()

# Initialize memory with an initial message from the assistant
initial_message = BaseAgentOutputSchema(chat_message="Hello! How can I assist you today?")
memory.add_message("assistant", initial_message)

# OpenAI client setup using the Instructor library
client = instructor.from_openai(
    openai.OpenAI(
        base_url="https://api.studio.nebius.com/v1",
        api_key=os.getenv("NEBIUS_API_KEY")
        )
    )

# Agent setup with specified configuration
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="Qwen/Qwen3-30B-A3B", 
        memory=memory,
    )
)

# Start a loop to handle user inputs and agent responses
while True:
    # Prompt the user for input
    user_input = console.input("[bold blue]You:[/bold blue] ")
    # Check if the user wants to exit the chat
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break
    
    memory.add_message("user", BaseAgentInputSchema(chat_message=user_input))
    # Process the user's input through the agent and get the response
    input_schema = BaseAgentInputSchema(chat_message=user_input)
    response = agent.run(input_schema)
    memory.add_message("assistant", response)
    # Display the agent's response
    console.print("Agent: ", response.chat_message)