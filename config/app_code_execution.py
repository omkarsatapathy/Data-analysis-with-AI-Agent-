import streamlit as st

class AppCodeExecution:
    """
    Manages the code execution result display.
    
    This class encapsulates UI elements for displaying the
    results of Python code execution.
    """
    
    @staticmethod
    def render_code_execution_results():
        """Render the results of code execution."""
        if st.session_state.code_execution_result:
            result = st.session_state.code_execution_result
            
            # Display figures (primary output)
            if result['figures']:
                st.subheader("Visualization Results")
                for i, fig in enumerate(result['figures']):
                    st.pyplot(fig)
            
            # Display text output if there are no figures or if there is output
            if result['output']:
                st.subheader("Analysis Results")
                st.text_area("", value=result['output'], height=300, disabled=True)