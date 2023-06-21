# Import necessary modules from the langchain package
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI

# Initialize a ChatOpenAI model instance with specified parameters
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Initialize an instance of the SerpAPIWrapper for web search
search = SerpAPIWrapper()

# Initialize an instance of LLMMathChain for mathematical operations
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# Define a list of tools the chatbot can use. Each tool has a name, 
# a function (the method that gets called when the tool is used), 
# and a description.
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events. You should ask targeted questions."
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math."
    )
]

# Initialize the agent with the specified tools and using OpenAI functions
mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

# Run the agent with a test query
mrkl.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
