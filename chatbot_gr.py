import os
import traceback
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

import gradio as gr

openai_api_key = "your key"

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, openai_api_key=openai_api_key)

loader = PyPDFDirectoryLoader("Company")
data = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=openai_embeddings)

retriever = vectorstore.as_retriever(k=8)

SYSTEM_TEMPLATE = """
You are a AI assistant.
Please answer the user's questions based on the below documents and the provided set of predefined answers for common questions.

If the question is technical and the context doesn't contain any relevant information, please just say "Sorry, I don't know" instead of making something up. 
For non-technical questions, refer to the predefined answers to provide accurate and helpful responses.

<context>
{context}
</context>

<examples>
{examples}
</examples>
"""

PARAPHRASE_QUESTION_TEMPLATE = """
You are a AI assistant. 
Here is a question which you don't know the answer based on the context: {question}. 
Please try to paraphrase this question into 3 different forms to make it more detailed based on the provided context and example questions and answers listed below. 

<context>
{context}
</context>

<examples>
{examples}
</examples>
"""

paraphrase_question_prompt = ChatPromptTemplate.from_messages([
    ('system', PARAPHRASE_QUESTION_TEMPLATE),
    MessagesPlaceholder(variable_name='question')
])

question_answering_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="messages"),
])

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)
paraphrase_question_chain = create_stuff_documents_chain(chat, paraphrase_question_prompt)

def create_few_shot_prompt(json_file):
    with open(json_file) as f:
        examples = json.load(f)
    fs_prompt = []
    for e in examples:
        fs_prompt.append(f"Question: {e['question']}\nAnswer: {e['answer']}")
    fs_prompt = "\n\n".join(fs_prompt)
    return fs_prompt

few_shot_prompt = create_few_shot_prompt("./fewshot.json")

def generate_response(question):
    try:
        context_docs = retriever.invoke(question)
        response = document_chain.invoke({
            "context": context_docs,
            "examples": few_shot_prompt,
            "messages": [HumanMessage(content=question)],
        })

        if response.startswith("Sorry, I don't know"):
            # Try to rephrase question to a generic open-ended question
            generic_question = f"What can you tell me about {question}?"
            generic_context_docs = retriever.invoke(question)
            response = document_chain.invoke({
                "context": generic_context_docs,
                "examples": few_shot_prompt,
                "messages": [HumanMessage(content=generic_question)],
            })

            if response.startswith("Sorry, I don't know"):
                response = paraphrase_question_chain.invoke({
                    "context": context_docs,
                    "examples": few_shot_prompt,
                    "question": [HumanMessage(content=question)]
                })
                response = f"Sorry, I don't quite understand your question. Can you be more specific about the question? For example:\n{response}"

        return response
    except Exception as e:
        traceback.print_exc()
        return "Error occurs!"

def save_history(username, history):
    filename = f"{username}_history.json"
    with open(filename, 'w') as f:
        json.dump(history, f)

def load_history(username):
    filename = f"{username}_history.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            history = json.load(f)
    else:
        history = []
    return history

def clear_history(username):
    filename = f"{username}_history.json"
    if os.path.exists(filename):
        os.remove(filename)

def my_chatbot(input, state):
    if state is None or not isinstance(state, dict):
        state = {"history": []}
    history = state.get("history", [])

    if len(history) == 0 or len(history[-1]) == 2:  # Start a new conversation if no ongoing conversation or the last one is complete
        history.append([])

    history[-1].append((input, ""))
    output = generate_response(input)
    history[-1][-1] = (input, output)
    save_history("admin", history)  # Save the history after each interaction
    state["history"] = history  # Save the updated history back to state
    chat_display = update_chat_display(history)  # Update chat display with timeline
    return chat_display, state

def update_chat_display(history):
    display = ""
    for i, conversation in enumerate(history):
        conversation_display = f"<div class='timeline'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
        for j, (user_input, response) in enumerate(conversation):
            conversation_display += f"<strong>Q{j+1}:</strong> {user_input}<br><strong>A{j+1}:</strong> {response}<br>"
        display += conversation_display
    return display

# Authentication function
def authenticate(username, password):
    if username == "admin" and password == "password":  # Example credentials
        return True
    else:
        return False

def login(username, password, state):
    if state is None or not isinstance(state, dict):
        state = {"history": []}
    if authenticate(username, password):
        history = load_history(username)
        state["history"] = history  # Save the loaded history to state
        chat_display = update_chat_display(history)  # Update chat display with timeline
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), chat_display, state
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, value="Invalid credentials, please try again."), state

with gr.Blocks(css="chatbot_gr.css") as demo:
    with gr.Column(elem_id="chat-container"):
        gr.Markdown("<div id='header'>Bidgely Chatbot</div>")
        
        # Define login and chat interfaces
        login_interface = gr.Column(visible=True)
        chat_interface = gr.Column(visible=False, elem_classes="chat-interface")
        
        # Login Interface
        with login_interface:
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Login")
            login_message = gr.Textbox(visible=False)
        
        # Chat Interface
        with chat_interface:
            chatbot = gr.HTML(elem_id="chat-history")
            state = gr.State(value={"history": []})
            with gr.Row(elem_classes="input-row"):
                txt = gr.Textbox(show_label=False, placeholder="Ask me a question and press enter.", elem_classes="input-box")
                send_btn = gr.Button("Send", elem_id="send-btn")
        
        # Callbacks and function definitions
        def clear_input():
            return ""

        login_button.click(login, inputs=[username, password, state], outputs=[chat_interface, login_interface, login_message, chatbot, state])
        username.submit(login, inputs=[username, password, state], outputs=[chat_interface, login_interface, login_message, chatbot, state])
        password.submit(login, inputs=[username, password, state], outputs=[chat_interface, login_interface, login_message, chatbot, state])
        send_btn.click(fn=my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
        txt.submit(fn=my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
        send_btn.click(fn=clear_input, inputs=None, outputs=txt)
        txt.submit(fn=clear_input, inputs=None, outputs=txt)

demo.launch(share=True, server_name='0.0.0.0', server_port=7860)
