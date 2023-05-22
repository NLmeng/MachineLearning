from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage, ServiceContext
from langchain.chat_models import ChatOpenAI
import gradio as gr

class Chatbot:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.storage_context = StorageContext.from_defaults(persist_dir="./storage")
        self.index = None
        self.max_input_size = 4096
        self.outputs = 512
        self.max_chunk_overlap = 20
        self.chunk_size_limit = 600

    def construct_new_index(self):
        # LLMPredictor (gpt-3.5-turbo) + ServiceContext
        prompt_helper = PromptHelper(self.max_input_size, self.num_outputs, self.max_chunk_overlap, chunk_size_limit=self.chunk_size_limit)
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=self.num_outputs))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)
        documents = SimpleDirectoryReader(self.directory_path).load_data()
        self.index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, prompt_helper=prompt_helper)
        self.index.storage_context.persist()

    def chatbot(self, input_text):
        if not self.index:
            self.index = load_index_from_storage(self.storage_context)
        query_engine = self.index.as_query_engine()
        response = query_engine.query(input_text)
        return response.response
    
    def create_interface(self):
        interface = gr.Interface(
            fn=self.chatbot,
            inputs=gr.components.Textbox(lines=7, label="Enter your text"),
            outputs="text",
            title="Personal AI Chatbot"
        )
        return interface
    
    def launch_interface(self):
        interface = self.create_interface()
        interface.launch(share=True)