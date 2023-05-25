import gradio as gr
from langchain.chat_models import ChatOpenAI
from llama_index import (GPTVectorStoreIndex, LLMPredictor, PromptHelper,
                         ServiceContext, SimpleDirectoryReader, StorageContext,
                         load_index_from_storage)


class ChatbotClass:
    def __init__(self, directory_path):
        """
        Initialize the ChatbotClass with a given directory path defaulting to /personalData.
        
        Args:
            directory_path (str): The path to the directory containing the data.
        """
        self.directory_path = directory_path
        self.storage_context = StorageContext.from_defaults(persist_dir="./storage")
        self.index = None
        self.max_input_size = 4096
        self.outputs = 512
        self.max_chunk_overlap = 20
        self.chunk_size_limit = 600

    def construct_new_index(self):
        """
        Construct a new index from the documents in the directory specified in the constructor.
        """
        # LLMPredictor (gpt-3.5-turbo) + ServiceContext
        prompt_helper = PromptHelper(self.max_input_size, self.num_outputs, self.max_chunk_overlap, chunk_size_limit=self.chunk_size_limit)
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=self.num_outputs))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)
        documents = SimpleDirectoryReader(self.directory_path).load_data()
        self.index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, prompt_helper=prompt_helper)
        self.index.storage_context.persist()

    def chatbot(self, input_text):
        """
        Generate a response to the given input text using the chatbot.
        
        Args:
            input_text (str): The input text to respond to.
        
        Returns:
            str: The chatbot's response.
        """
        if not self.index:
            self.index = load_index_from_storage(self.storage_context)
        query_engine = self.index.as_query_engine()
        response = query_engine.query(input_text)
        return response.response
    
    def launch_interface(self):
        """
        Create and then launch the Gradio interface for the chatbot.
        """
        def create_interface():
            interface = gr.Interface(
                fn=self.chatbot,
                inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                outputs="text",
                title="Personal AI Chatbot"
            )
            return interface
        interface = create_interface()
        interface.launch(share=True)

# export and use this instance
Chatbot = ChatbotClass("personalData")