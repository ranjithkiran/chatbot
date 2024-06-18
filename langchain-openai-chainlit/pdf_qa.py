# Import necessary modules and define env variables

import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl


# text_splitter and system template

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# system_template = """Use the following pieces of context to answer the users question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# ALWAYS return a "SOURCES" part in your answer.
# The "SOURCES" part should be a reference to the source of the document from which you got your answer.

# Example of your response should be:

# ```
# The answer is foo
# SOURCES: xyz
# ```

# Begin!
# ----------------
# {summaries}"""


# messages = [
#     SystemMessagePromptTemplate.from_template(system_template),
#     HumanMessagePromptTemplate.from_template("{question}"),
# ]
# prompt = ChatPromptTemplate.from_messages(messages)
# chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
async def on_chat_start():

    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!", elements=elements).send()
    files = None

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=18000,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the pdf file
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    # Create a chain that uses Chroma Vector Space 
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama3"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    # store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    print("message")
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 

    # Asynchronous callback
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with message content
    res = await chain.ainvoke(message.content, callbacks=[cb], show_progress=True)
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []

    # Process source documents if available
    if source_documents:
        # Add the sources to the message
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add Source references to answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    await cl.Message(content=answer, elements=text_elements).send()