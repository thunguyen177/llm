#pip install crewai
#pip install -U langchain-ollama
#ollama pull llama3
from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
# Dòng này yêu cầu tất cả các agent sử dụng mô hình Llama3 cục bộ của bạn
ollama_llm = OllamaLLM(model="ollama/llama3")

# Agent 1: Nhà nghiên cứu
researcher = Agent(
  role='Researcher', # Vai trò
  goal='Research a new topic', # Mục tiêu
  backstory='You are a master of desk research.', # Tiểu sử
  llm=ollama_llm,
  allow_delegation=False # Agent này làm việc một mình
)
# Agent 2: Người viết
writer = Agent(
  role='Writer', # Vai trò
  goal='Write a compelling article', # Mục tiêu
  backstory='You are a renowned tech writer.', # Tiểu sử
  llm=ollama_llm,
  allow_delegation=False # Agent này cũng làm việc một mình
)

# Tác vụ 1: Giao cho nhà nghiên cứu
research_task = Task(
  description='Investigate the latest advancements in AI.', # Mô tả
  expected_output='A summary of the top 3 advancements.', # Đầu ra mong đợi
  agent=researcher # Tác vụ này PHẢI được thực hiện bởi agent 1: researcher
)
# Tác vụ 2: Giao cho người viết
writing_task = Task(
  description='Based on the research, write an article.', # Mô tả
  expected_output='A 500-word article on the latest AI advancements.', # Đầu ra mong đợi
  agent=writer # Tác vụ này PHẢI được thực hiện bởi agent 2: người viết 
)

# Crew bao gồm cả hai agent và các tác vụ được giao của họ
crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, writing_task],
  verbose=True
)
# Lệnh duy nhất này sẽ bắt đầu toàn bộ luồng công việc đa agent
result = crew.kickoff()

print(result)