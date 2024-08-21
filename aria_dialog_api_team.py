from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

class ChatCSV:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        """
        Initializes the question-answering system with default configurations.
        This constructor sets up the following components:
        - A ChatOllama model for generating responses ('neural-chat').
        - A RecursiveCharacterTextSplitter for splitting text into chunks.
        - A PromptTemplate for constructing prompts with placeholders for question and context.
        """
        # Initialize the ChatOllama model with 'neural-chat'.
        self.model = ChatOllama(base_url = 'http://54.255.10.70:11434', model= 'llama2')

        # Initialize the RecursiveCharacterTextSplitter with specific chunk settings.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        # Initialize the PromptTemplate with a predefined template for constructing prompts.
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are a helpful nutritionist that analyses different ingredients to come up with a meal plan.
            Use the following pieces of retrieved context to answer the question.
            Give Name when possible. If you don't know the answer,
            just say that you don't know.  [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        
    def ingest(self, csv_file_path: str):
        '''
        Ingests data from a CSV file containing resumes, process the data, and set up the
        components for further analysis.
        Parameters:
        - csv_file_path (str): The file path to the CSV file.
        Usage:
        obj.ingest("/path/to/data.csv")
        This function uses a CSVLoader to load the data from the specified CSV file.
        Args:
        - file.path (str): The path to the CSV file.
        - encoding (str): The character encoding of the file (default is 'utf-8').
        - source_column (str): The column in the CSV containing the data (default is "Resume").
        '''        
        loader = CSVLoader(
            file_path=csv_file_path,
            # file_path = "D:\Python Projects\RAG_LLM\Menu.csv"
            encoding='utf-8',
            # source_column="Resume"
            csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['NAME','COST','Kcal','Fat','Carb','Protein','Fiber','MealSize','Ingrediants Needed']
            }
            )
        
        # loads the data
        data = loader.load()

        # splits the documents into chunks
        chunks = self.text_splitter.split_documents(data)
        chunks = filter_complex_metadata(chunks)

        # creates a vector store using embedding
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # sets up the retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # Define a processing chain for handling a question-answer scenario.
        # The chain consists of the following components:
        # 1. "context" from the retriever
        # 2. A passthrough for the "question"
        # 3. Processing with the "prompt"
        # 4. Interaction with the "model"
        # 5. Parsing the output using the "StrOutputParser"
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
    def ask(self, query: str):
        """
        Asks a question using the configured processing chain.
        Parameters:
        - query (str): The question to be asked.
        Returns:
        - str: The result of processing the question through the configured chain.
        If the processing chain is not set up (empty), a message is returned
        prompting to add a CSV document first.
        """
        if not self.chain:
            return "Please, add a CSV document first."

        return self.chain.invoke(query)

    def clear(self):
        """
        Clears the components in the question-answering system.
        This method resets the vector store, retriever, and processing chain to None,
        effectively clearing the existing configuration.
        """
        # Set the vector store to None.
        self.vector_store = None

        # Set the retriever to None.
        self.retriever = None

        # Set the processing chain to None.
        self.chain = None

#=========================modified app.py===================================




#========================Aria original code=========================
from utils import convert_text_to_html
from aria_dialog_api_base import AriaDialogAPI

class Team_ARIADialogAPI(AriaDialogAPI):
    """This is a simple example of a class that inherits AriaDialogAPI and implements the ARIA
    dialog API for a toy model that echos back whatever prompt is given."""
    model_name = 'Echo'
    def OpenConnection(self, auth=None):
        """Opens a connection to the application.

        The parameter `auth` is ignored.

        Parameters
        ----------
        auth : dict, optional
            the authorization dictionary that, for example, might contain an API key as a value;
            since there is no authorization needed for the Echo model, the parameter is ignored.

        Returns
        -------
        bool
            a boolean value indicating whether the connection was successfully opened; since there
            is no connection made for the Echo mode, simply returns True.
        """
        return True


    def CloseConnection(self):
        """Closes an open connection to the application.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            a boolean value indicating whether the connection was successfully opened. Since there
            is no connection made for the Echo mode, simply returns True.
        """
        return True
    @staticmethod
    def GetVersion():
        """Returns the version of the API implementation.

        Parameters
        ----------
        None

        Returns
        -------
        string
            a string indicating the version of the API implementation.
        """
        return '0.1'
    def StartSession(self):
        """Starts a new dialog session.

       Parameters
       ----------
       None

       print ("start chatbot", response_chatbot)
       

       Returns
       -------
       bool
           a boolean value indicating whether the session was successfully started.
       """


        self.chatbot = ChatCSV()
        print ("before ingest ============================================")
        self.chatbot.ingest ("menu.csv")
        print ("after ingest ============================================")

        #st.session_state["assistant"] = ChatCSV()
        #st.session_state["assistant"].ingest 
        return True
    def GetResponse(self, text):
        """Returns a response to text prompt.

         Parameters
         ----------
         text : str
             The prompt from the user to be provided to the model

         Returns
         -------
         dictionary
             a dictionary with keys "success" and "response" with values indicating whether the app
             successfully returned a response and the response itself in html format, respectively.
         """
        print ("In get response")
         
        response_chatbot = self.chatbot.ask (text)
        print ("response chatbot", response_chatbot)


        htmltext = convert_text_to_html(text)
        htmltext = response_chatbot
        return {'success': True,
                'response': htmltext}
