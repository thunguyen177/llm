#pip install crewai
#pip install -U langchain-ollama
#ollama pull llama3
from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM

ollama_llm = OllamaLLM(model="ollama/llama3")

# Define your agents
researcher = Agent(
  role='Researcher',
  goal='Research a new topic',
  backstory='You are a master of desk research.',
  llm=ollama_llm,
  allow_delegation=False
)

writer = Agent(
  role='Writer',
  goal='Write a compelling article',
  backstory='You are a renowned tech writer.',
  llm=ollama_llm,
  allow_delegation=False
)

# Define your tasks
research_task = Task(
  description='Investigate the latest advancements in AI.',
  expected_output='A summary of the top 3 advancements.',
  agent=researcher
)

writing_task = Task(
  description='Based on the research, write an article.',
  expected_output='A 500-word article on the latest AI advancements.',
  agent=writer
)

crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, writing_task],
  verbose=True # Correct: uses a boolean
)

result = crew.kickoff()

print(result)