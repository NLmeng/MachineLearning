from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage, ServiceContext
from langchain.chat_models import ChatOpenAI
import gradio as gr

# run export OPENAI_API_KEY='..'
# python 3.11.2
# pip 22.3.1
# llama-index 0.5.4
# PyPDF2 3.0.1
# gradio 3.24.1
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.storage_context.persist()

    return index

def construct_index_new(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    #
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    # LLM Predictor (gpt-3.5-turbo) + service context
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)
    #
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, prompt_helper=prompt_helper)
    return index

def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

if __name__ == '__main__':
    interface = gr.Interface(
        fn=chatbot,
        inputs=gr.components.Textbox(lines=7, label="Enter your text"),
        outputs="text",
        title="Personal AI Chatbot"
    )
    index = construct_index("personalData")
    # index = construct_index_new("personalData")
    interface.launch(share=True)