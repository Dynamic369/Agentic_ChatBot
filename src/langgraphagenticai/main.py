import streamlit as st
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit


def load_langgraph_agentic_app():
    """
    Loads and run the Langgraph AgenticAI Application with streamlit UI.
    This function is to initialize the UI , handles the user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while
    implementing exception handling for robustness.
    """

    #Load the Ui
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.warning("Error: Failed to load the user input from the UI.")
        return
    
    user_message = st.chat_input("Enter your message")

    if user_message:
        try:
            #Configure the llm
            obj_config_llm = GroqLLM(user_controls_input=user_input)
            model = obj_config_llm.get_llm_model()

            if not model:
                st.error("Error llm model could not be initialized")
                return
            
            #Initialize and setup the graph based on use case
            usecase = user_input.get('selected_usecase')

            if not usecase:
                st.error("Error: No usecase selected.")
                return
            
            ## Grpah builder
            graph_builder = GraphBuilder(model)
            try:
                graph = graph_builder.setup_graph(usecase)
                DisplayResultStreamlit(usecase,graph,user_message).display_result_on_ui()

            except Exception as e:
                st.error(f"Error: Graph setup failed {e}")
                return
        
        except Exception as e:
            st.error(f"Error: Graph setup failed {e}")
            return

        



