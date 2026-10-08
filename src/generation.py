import os
from typing import List, Dict, Any, Union, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging

# level of logger

# DUBUG
# INFO
# WARING
# ERROR
# CRITICAL

logging.basicConfig(filename="log_tracker.text",
                    filemode= "a",
                    level = logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%y-%m-%d %H %M %S")


class LLMModelNotLoaded(Exception):
    pass




class LLM:

    def __init__(self, 
                groq_api_key: str = "GROQ_API_KEY",
                model_name: str = "openai/gpt-oss-120b",
                temperature: float = 0):


        try:

            self.stroutputparser = StrOutputParser()
            load_dotenv()
            get_apikey = os.getenv(groq_api_key)

            if not get_apikey:
                logging.error("No groq api-key found in .env file. Please set your groq api-key in .env")
                raise ValueError("No groq api-key found in .env file. Please set your groq api-key in .env")

            self.llm_model = ChatGroq(model = model_name,
                                      temperature = temperature,
                                      api_key = get_apikey)
            logging.info("LLM model have loaded")

        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e
            
    def llm_Generator(self, query: str, document: List[str]):

        RAG_SYSTEM_PROMPT = """You are a factual assistant that answers questions strictly based on the provided context.
        RULES:
        1. Answer ONLY using information present in the CONTEXT below. Do not use any outside knowledge.
        2. If the context does not contain enough information to answer the question, respond with:
        "I don't have enough information in the provided context to answer this question."
        3. Do not make assumptions, do not guess, and do not fill gaps with general knowledge about the topic.
        4. If multiple context chunks provide related information, synthesize them into one clear, coherent answer — do not just list chunks separately.
        5. Keep the answer concise and directly relevant to the question. Avoid repeating the same information twice.
        6. Do not mention "context", "chunks", "documents", or "retrieval" in your answer — respond naturally as if you already know this information.
        7. If numbers, dates, or figures are present in the context, use them exactly as given — do not round, estimate, or alter them.
        8. Do not add disclaimers like "based on the text provided" — just answer directly.
        
        """

        try:
            if not self.llm_model:
                logging.error("LLM model not loaded may no llm api-key or else give proper LLM model name")
                raise LLMModelNotLoaded("LLM model not loaded may no llm api-key or else give proper LLM model name")

            prompt_template = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT),
                    HumanMessagePromptTemplate.from_template("Question: {query}\n\nContext:\n{context}\n\nAnswer:")
                ])
                

            chain = prompt_template | self.llm_model | self.stroutputparser
            logging.info("chain have started")

            generator_final_result = chain.invoke({"query":query,"context":document})
            logging.info("finally got the answer from llm")

            return generator_final_result

        except Exception as e:
            logging.error(f"ERROR {e}")
            raise e