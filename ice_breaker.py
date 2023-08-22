import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.tweeter import scrape_user_tweets
from agents.linkedin_lookup_agent import lookup_profile
from output_parser import person_intel_parser, PersonIntel

load_dotenv()


def ice_breaker(name: str) -> tuple[PersonIntel, str]:
    linkedin_profile_url = lookup_profile(name=name)
    # print(os.environ["OPENAI_API_KEY"])

    summary_templet = """
        given the information {information} about a person from i want you to create:
        1. a short summary
        2. two interesting facts about them
        3. A topic that may interest them
        4. 2 creative Ice breakers to open a conversation with them 
                \n{format_instructions}
     """
    summary_prompt_tamplet = PromptTemplate(
        input_variables=["information"],
        template=summary_templet,
        partial_variables={
            # get_format_instructions() gets the output parser we created and extract the schema and plug it into the template
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_tamplet)
    data = scrape_linkedin_profile(linkedin_profile_url)
    result = chain.run(information=data)
    print(data.get("profile_pic_url"))
    return person_intel_parser.parse(result), data.get("profile_pic_url")
    # print(scrape_user_tweets(username="@imVkohli", num_tweets=100))


if __name__ == "__main__":
    print("Hello Langchain")
    result = ice_breaker(name="Ashneer Grover BharatPe")
