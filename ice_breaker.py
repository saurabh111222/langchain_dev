import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup_profile
load_dotenv()

if __name__ == "__main__":
    print("Hello Langchain")

    linkedin_profile_url = lookup_profile(name = "Saurabh Kumar iNeuron IIT Dhanbad")
    # print(os.environ["OPENAI_API_KEY"])

    summary_templet = """
        given the information {information} about a person from i want you to create:
        1. a short summary
        2. two interesting facts about them
"""
    summary_prompt_tamplet = PromptTemplate(
        input_variables=["information"], template=summary_templet
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_tamplet)
    data = scrape_linkedin_profile(linkedin_profile_url)
    print(chain.run(information=data))