import chainlit as cl
import os
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import json
from pathlib import Path


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API key
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

# Initialize Groq client for testing
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    # Test the API key with a simple request
    groq_client.chat.completions.create(
        messages=[{"role": "user", "content": "test"}],
        model="llama-3.1-8b-instant",
        max_tokens=10
    )
    print("API key validation successful!")
except Exception as e:
    print(f"Error validating API key: {str(e)}")
    raise ValueError(f"Invalid GROQ_API_KEY or API error: {str(e)}")

# Initialize different models with error handling
try:
    LLAMA_MODEL = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        max_tokens=2048,
        temperature=0.7
    )
    MISTRAL_MODEL = ChatGroq(
        model="mistral-7b-8192",
        api_key=GROQ_API_KEY,
        max_tokens=2048,
        temperature=0.7
    )
    GEMMA_MODEL = ChatGroq(
        model="gemma2-9b-it",
        api_key=GROQ_API_KEY,
        max_tokens=2048,
        temperature=0.7
    )
    EVALUATOR_MODEL = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        max_tokens=2048,
        temperature=0.7
    )
    print("All models initialized successfully!")
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise

# Create memory storage
MEMORY_FILE = "chat_memory.json"
memory = ConversationBufferMemory()

def load_memory():
    if Path(MEMORY_FILE).exists():
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {"history": []}

def save_memory(memory_data):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory_data, f)

# Prompt templates
BASE_PROMPT = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI assistant. Provide detailed and accurate responses."),
        ("user", "{text}"),
    ]
)

EVALUATOR_PROMPT = ChatPromptTemplate(
    [
        ("system", """You are an AI evaluator. Your task is to:
        1. Analyze the responses from different AI models
        2. Identify the strengths and weaknesses of each response
        3. Combine the best elements into a comprehensive answer
        4. Provide a final, well-structured response
        
        The responses to evaluate are:
        {responses}
        
        Please provide your evaluation and combined response."""),
        ("user", "{text}"),
    ]
)

PARSER = StrOutputParser()

@cl.on_chat_start
async def start():
    await cl.Message(content="Merhaba! I'm your AI assistant. I'll combine insights from multiple AI models to provide you with the best possible response. What would you like to know?").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Load previous memory
        memory_data = load_memory()
        
        # Get responses from different models
        llama_chain = BASE_PROMPT | LLAMA_MODEL | PARSER
        mistral_chain = BASE_PROMPT | MISTRAL_MODEL | PARSER
        gemma_chain = BASE_PROMPT | GEMMA_MODEL | PARSER
        
        # Get responses from all models with error handling
        try:
            llama_response = await llama_chain.ainvoke({"text": message.content})
        except Exception as e:
            llama_response = f"Error with Llama model: {str(e)}"
            print(f"Llama model error: {str(e)}")
            
        try:
            mistral_response = await mistral_chain.ainvoke({"text": message.content})
        except Exception as e:
            mistral_response = f"Error with Mistral model: {str(e)}"
            print(f"Mistral model error: {str(e)}")
            
        try:
            gemma_response = await gemma_chain.ainvoke({"text": message.content})
        except Exception as e:
            gemma_response = f"Error with Gemma model: {str(e)}"
            print(f"Gemma model error: {str(e)}")
        
        # Combine responses for evaluation
        combined_responses = f"""
        Llama Response: {llama_response}
        Mistral Response: {mistral_response}
        Gemma Response: {gemma_response}
        """
        
        # Evaluate and combine responses
        try:
            evaluator_chain = EVALUATOR_PROMPT | EVALUATOR_MODEL | PARSER
            final_response = await evaluator_chain.ainvoke({
                "text": message.content,
                "responses": combined_responses
            })
        except Exception as e:
            final_response = f"Error with evaluator model: {str(e)}"
            print(f"Evaluator model error: {str(e)}")
        
        # Update memory
        memory_data["history"].append({
            "user": message.content,
            "response": final_response
        })
        save_memory(memory_data)
        
        # Send the final response
        await cl.Message(content=final_response, author="AI Assistant").send()
        
        # Send individual model responses for transparency
        await cl.Message(content=f"Llama: {llama_response}", author="Llama").send()
        await cl.Message(content=f"Mistral: {mistral_response}", author="Mistral").send()
        await cl.Message(content=f"Gemma: {gemma_response}", author="Gemma").send()
        
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        await cl.Message(content=error_message, author="System Error").send()

@cl.on_chat_end
async def on_chat_end():
    # Delete the memory file if it exists
    if Path(MEMORY_FILE).exists():
        try:
            os.remove(MEMORY_FILE)
            print(f"Chat memory file {MEMORY_FILE} has been deleted.")
        except Exception as e:
            print(f"Error deleting memory file: {e}")

