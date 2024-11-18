import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import IPython.display
import io
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from database.create_db import create_db_info
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
import re
# Import dotenv functions
# dotenv allows you to read environment variables from a .env file
# This is especially useful in development to avoid hardcoding sensitive information (e.g., API keys) into the code

# Locate the .env file and load its contents
# This allows you to use os.environ to read environment variables set in the .env file
_ = load_dotenv(find_dotenv())
LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
}

LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
INIT_LLM = "chatglm_std"
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge_db')
DEFAULT_PERSIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_db')
USER_AVATAR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'user.png')
CHATBOT_AVATAR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'robot.png')

def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")

class Model_center():
    """
    Stores objects of QA chains 

    - chat_qa_chain_self: Stores QA chains with history, with (model, embedding) as keys.
    - qa_chain_self: Stores QA chains without history, with (model, embedding) as keys.
    """
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "openai", embedding: str = "openai", temperature: float = 0.0, top_k: int = 4, history_len: int = 3, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH):
        """
        Uses the QA chain with history to answer questions
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(model=model, temperature=temperature,
                                                                                    top_k=top_k, chat_history=chat_history, file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.chat_qa_chain_self[(model, embedding)]
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            return e, chat_history

    def qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "openai", embedding="openai", temperature: float = 0.0, top_k: int = 4, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH):
        """
        Uses the QA chain without history to answer questions
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = QA_chain_self(model=model, temperature=temperature,
                                                                       top_k=top_k, file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.qa_chain_self[(model, embedding)]
            chat_history.append(
                (question, chain.answer(question, temperature, top_k)))
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()


def format_chat_prompt(message, chat_history):
    """
    Formats the chat prompt.

    Parameters:
    message: The current user message.
    chat_history: Chat history.

    Returns:
    prompt: Formatted prompt.
    """
    # Initialize an empty string for the formatted chat prompt.
    prompt = ""
    # Traverse chat history.
    for turn in chat_history:
        # Extract user and bot messages from chat history.
        user_message, bot_message = turn
        # Update prompt by adding user and bot messages.
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # Add the current user message to the prompt, leaving space for bot response.
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # Return the formatted prompt.
    return prompt


def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    """
    Generates the bot's response.

    Parameters:
    message: The current user message.
    chat_history: Chat history.

    Returns:
    "": An empty string means nothing needs to be displayed on the screen, replace with actual bot response if needed.
    chat_history: Updated chat history
    """
    if message == None or len(message) < 1:
        return "", chat_history
    try:
        # Limit history memory length
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # Use the function above to format the user message and chat history into a prompt.
        formatted_prompt = format_chat_prompt(message, chat_history)
        # Generate bot's response using the predict method of the llm object (note: llm object is not defined in this code).
        bot_message = get_completion(
            formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
        # Replace \n with <br/> in bot_message
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        # Add user message and bot response to chat history.
        chat_history.append([message, bot_message])
        # Return an empty string and updated chat history (replace the empty string with actual bot response if needed for display).
        return "", chat_history
    except Exception as e:
        return e, chat_history


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):           
   
        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>Personal Knowledge Base Assistant</center></h1>
                <center>by Fengli Wu</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=480, show_copy_button=True, show_share_button=True, avatar_images=(USER_AVATAR_PATH, CHATBOT_AVATAR_PATH))
            # Create a text box component for entering the prompt.
            msg = gr.Textbox(label="Prompt")

            with gr.Row():
                # Create submit buttons.
                db_with_his_btn = gr.Button("Chat with DB & history")
                db_wo_his_btn = gr.Button("Chat with DB only")
                llm_btn = gr.Button("Chat with LLM only")
            with gr.Row():
                # Create a clear button to clear the chatbot component content.
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        with gr.Column(scale=1):
            file = gr.File(label='Select file directory', file_count='directory',
                           file_types=['.txt', '.md', '.docx', '.pdf'])
            with gr.Row():
                init_db = gr.Button("File vectorization")
            model_argument = gr.Accordion("Parameters setting", open=False)
            with model_argument:
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="LLM temperature",
                                        interactive=True)

                top_k = gr.Slider(1,
                                  10,
                                  value=3,
                                  step=1,
                                  label="Vector DB search top k",
                                  interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="History length",
                                        interactive=True)

            model_select = gr.Accordion("Model selection", open=False)
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="Language model",
                    value=INIT_LLM,
                    interactive=True)

                embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                         label="Embedding model",
                                         value=INIT_EMBEDDING_MODEL)

        # Set the initialization vector database button's click event. Calls create_db_info when clicked, passing in the user's file and chosen Embedding model.
        init_db.click(create_db_info,
                      inputs=[file, embeddings], outputs=[msg])

        # Set button click event. Calls chat_qa_chain_self_answer function when clicked, passing in user's message and chat history, then updates the textbox and chatbot component.
        db_with_his_btn.click(model_center.chat_qa_chain_self_answer, inputs=[
                              msg, chatbot,  llm, embeddings, temperature, top_k, history_len],
                              outputs=[msg, chatbot])
        # Set button click event. Calls qa_chain_self_answer function when clicked, passing in user's message and chat history, then updates the textbox and chatbot component.
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot, llm, embeddings, temperature, top_k], outputs=[msg, chatbot])
        # Set button click event. Calls respond function when clicked, passing in user's message and chat history, then updates the textbox and chatbot component.
        llm_btn.click(respond, inputs=[
                      msg, chatbot, llm, history_len, temperature], outputs=[msg, chatbot], show_progress="minimal")

        # Set text box submission event (when Enter key is pressed). Functionality is the same as the llm_btn button click event.
        msg.submit(respond, inputs=[
                   msg, chatbot,  llm, history_len, temperature], outputs=[msg, chatbot], show_progress="hidden")
        # Click to clear stored chat history
        clear.click(model_center.clear_history)
    # gr.Markdown("""Reminder:<br>
    # """)
# threads to consume the request
gr.close_all()
# Start a new Gradio app, set sharing to True, and use environment variable PORT1 to specify server port.
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# Launch directly
demo.launch()

