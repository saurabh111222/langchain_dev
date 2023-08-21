from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain import PromptTemplate
from tools.tools import get_profile_url

def lookup_profile(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    templet = """given the full name {name_of_person} I want you to get me the link to their LinkedIn profile page.
                Your answer should contain only a URL"""
    
    tools_for_agent=[
        Tool(
        name = "Crawl google for a linkedin page",
        func=get_profile_url,
        description="useful for when you need to get the LinkedIn page URL"
        )
    ]

    agent = initialize_agent(tools=tools_for_agent, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)

    prompt_templet = PromptTemplate(template=templet, input_variables=['name_of_person'])

    linkedin_page_url = agent.run(prompt_templet.format_prompt(name_of_person=name))

    return linkedin_page_url