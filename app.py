import streamlit as st
import pandas as pd
import plotly.express as px
from utils import *
from transformers import pipeline

from keras.models import load_model


model = load_model('domestic_best_model.keras')


def load_model_pipeline(model_id):
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",  # Use automatic precision
        device_map="auto",  # Automatically map to available GPUs/CPUs
    )


if "local_chat" not in st.session_state:
    st.session_state.local_chat = [
        {"role": "system", "content": conversation_system_prompt},
    ]
if "response_generated" not in st.session_state:
    st.session_state.response_generated = False  # Initialize the flag
if "page" not in st.session_state:
    st.session_state.page = 1
# Initialize session state variables
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "data" not in st.session_state:
    st.session_state.data = None
if "price_per_litre" not in st.session_state:
    st.session_state.price_per_litre = 0.0
if "max_budget" not in st.session_state:
    st.session_state.max_budget = 0.0
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": conversation_system_prompt}
    ]
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
st.set_page_config(page_title="AquaQuest AI", page_icon="🌊")
# First Page: Model Selection
def model_selection_page():
    st.title("💧 Welcome to AquaQuest AI")
    st.write("Model defined at code the line 76 of the program. You can view the model's description and requirements before proceeding.")

    # Define available models
    models = {
        "Llama 3.2 - 1B": {
            "id": "meta-llama/Llama-3.2-1B-Instruct",
            "description": "This is the smallest model in the Llama 3.2 lineup, optimized for lightweight tasks. It is capable of handling basic text generation and multilingual dialogue use cases. While it is not as powerful as larger models, it is highly efficient and suitable for applications where computational resources are limited. Not recommended for this application, but you may be able to try it.",
            "requirements": {
                "GPU": "NVIDIA RTX 3060 or higher",
                "VRAM": "8 GB",
            },
        },
        "Llama 3.2 - 3B": {
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "description": "LLaMA 3.2 3B is a lightweight model with strong summarization capabilities and moderate abilities in reasoning and comprehension.",
            "requirements": {
                "GPU": "NVIDIA RTX 4080 or higher",
                "VRAM": "15 GB",
            },
        },"Llama 3.1 - 8B": {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "description": "LLaMA 3.1 8B is the smallest multimodal large language model developed by Meta. It possesses moderate capabilities in reasoning and creativity, along with the ability to process and generate content across multiple modalities effectively.",
            "requirements": {
                "GPU": "NVIDIA A40 or higher",
                "VRAM": "40 GB",
            },
        },
    }

    selected_model_name = "Llama 3.2 - 1B"  # Llama 3.2 - 1B or Llama 3.2 - 3B or Llama 3.1 - 8B
    st.header(selected_model_name)
    # Display model details dynamically
    if selected_model_name:
   
        selected_model = models[selected_model_name]

        # Create a 2x2 layout
        col1, col2 = st.columns(2)
        # Top row: Model Description and Requirements
        with col1:
            st.subheader(f"📄 Model Description")
            st.write(selected_model["description"])

        with col2:
            st.subheader("⚙️ Requirements")
            requirements = selected_model["requirements"]
            st.write(f"**GPU:** {requirements['GPU']}")
            st.write(f"**VRAM:** {requirements['VRAM']}")

                # Bottom row: Instructions and Version Notes
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📋 Instructions")
            st.write("""
            1. Select a model at the line 76 of the app.py file.
            2. Review the model's description and requirements.
            3. Click the **Introduce Data** button to load the model.
            4. Upload a csv file with the data.:
                - The CSV file have to contain data of the daily usage of water for at least 360 days.
                - The CSV file have to contain data as the Y row the time.
                - The CSV file have to contain data as the X row the usage of water.
            5. Input the price per litre and the maximum budget for the next 30 days.
            6. Click the **Chat with AquaQuest AI** button to start the chatbot.
            7. Ask questions and interact with the chatbot.
            **NOTE**: The model will take some time to load which depends on the power of your Hardware. Please be patient.
            """)

        with col4:
            st.subheader("📝 Version Notes")
            st.write("""
            - **Version Beta**: Initial release of the chatbox application.
            - **Further Version**: Add accurate models for industial and commercial use.
            """)



        # Button to navigate to the data introduction page
        if st.button("Introduce Data"):
            st.session_state.selected_model = selected_model
            st.session_state.page = 2
            st.experimental_rerun()

# Second Page: Data Introduction
def data_introduction_page():
    st.title("📊 Upload Data and Create Predictions")
    st.write("Upload your data of at least 360 days, visualize the last 150 days, and predict the price for the next 30 days.")

    # Step 1: Upload CSV File
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df  # Save the data in session state
            st.success("✅ File uploaded successfully!")

        except Exception as e:
            st.error(f"Error processing the file: {e}")

    # Check if data is already uploaded
    if st.session_state.data is not None:
        df = st.session_state.data

        print(df)

        # Step 2: Plot the Last 150 Days (Interactive Plot)
        st.subheader("📈 Last 150 Days of Data")
        last_150_days = get_last_days(df)
        last_150_days_df = pd.DataFrame({"Day": range(1, len(last_150_days) + 1), "Value": last_150_days})
        fig = px.line(last_150_days_df, x="Day", y="Value", title="Last 150 Days of Data", markers=True)
        st.plotly_chart(fig)

        # Step 3: Input Budget and Price per Litre
        st.subheader("💰 Budget and Price Inputs")
        price_per_litre = st.number_input("Price per litre (€)", min_value=0.0, step=0.01, format="%.2f", key="price_input")
        max_budget = st.number_input("Maximum budget for the next 30 days (€)", min_value=0.0, step=0.01, format="%.2f", key="budget_input")

        # Save budget and price in session state
        st.session_state.price_per_litre = price_per_litre
        st.session_state.max_budget = max_budget

        # Step 4: Create Predictions
        if price_per_litre > 0 and max_budget > 0:
            st.subheader("🔮 Predictions for the Next 30 Days")
            predictions = create_predictions(model=model, dataset=df)  # Replace `None` with your model
            st.session_state.predictions = predictions  # Save predictions in session state

            predicted_price = float(predictions["30_days"]) * price_per_litre
            st.write(f"**Predicted Consumption for the Next Day:** {predictions['1_day']} litres")
            st.write(f"**Predicted Consumption for the Next 15 Day:** {predictions['15_days']} litres")
            st.write(f"**Predicted Consumption for the Next 30 Days:** {predictions['30_days']} litres")
            st.write(f"**Predicted Total Cost:** €{predicted_price:.2f}")


        go_to_chatbot = st.button("Chat with AquaQuest AI")  # Button to navigate to chatbot

        # Only navigate to chatbot if the button is pressed and inputs are valid
        if go_to_chatbot:
            if price_per_litre > 0 and max_budget > 0:
                if not st.session_state.model_loaded:
                    with st.spinner("Loading the model (This can take a long time)..."):
                        # Load the selected model
                        st.session_state.llama_pipeline = load_model_pipeline(
                            st.session_state.selected_model["id"]
                        )

                    st.session_state.model_loaded = True  # Mark the model as loaded
                st.session_state.page = 3

                st.experimental_rerun()  # Navigate to the chatbot page

            else:
                st.error("⚠️ Please enter a valid price per litre and budget to proceed.")

def chatbot_page():
    st.title("💬 Chat with AquaQuest AI")
    st.write("Interact with AquaQuest AI to get insights and recommendations based on your data.")

    # Sidebar: Display uploaded data, budget, and predictions
    with st.sidebar:
        st.header("📄 YOUR DATA")
        df = st.session_state.data
        last_150_days = get_last_days(df)
        last_150_days_df = pd.DataFrame({"Day": range(1, len(last_150_days) + 1), "Value": last_150_days})
        fig = px.line(
            last_150_days_df,
            x="Day",
            y="Value",
            title="Last 150 Days of Data",
            markers=True,
            width=300,  # Adjust width to fit the sidebar
            height=300,  # Adjust height to fit the sidebar
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💰 Budget and Price Inputs")
        st.write(f"**Price per litre:** €{st.session_state.price_per_litre:.2f}")
        st.write(f"**Maximum budget:** €{st.session_state.max_budget:.2f}")

        st.subheader("🔮 Predictions")
        if st.session_state.predictions is not None:
            st.write(f"**Predicted Consumption (1 Day):** {st.session_state.predictions['1_day']} litres")
            st.write(f"**Predicted Consumption (15 Days):** {st.session_state.predictions['15_days']} litres")
            st.write(f"**Predicted Consumption (30 Days):** {st.session_state.predictions['30_days']} litres")
            predicted_price = float(st.session_state.predictions["30_days"]) * st.session_state.price_per_litre
            st.write(f"**Predicted cost Next 30 days:** €{predicted_price:.2f}")
        
        # Button to start a new session
        if st.button("🔄 New Session"):
            st.session_state.page = 1

            # Clean all the variables
            st.session_state.data = None	
            st.session_state.price_per_litre = 0.0
            st.session_state.max_budget = 0.0
            st.session_state.predictions = None
            st.session_state.chat_history = [
                {"role": "system", "content": conversation_system_prompt}
            ]
            st.session_state.local_chat = [
                {"role": "system", "content": conversation_system_prompt},
            ]
            st.session_state.response_generated = False  # Reset the fla
            st.experimental_rerun()


    for message in st.session_state.local_chat[1:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    if not st.session_state.response_generated:
        # Set waiting_for_response to True to disable furt
        # her input
        st.session_state.waiting_for_response = True

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("AquaQuest AI is studying your data..."):
                start_prompt, input_str = create_start_prompt(
                    st.session_state.llama_pipeline,
                    st.session_state.data,
                    st.session_state.max_budget,
                    st.session_state.predictions,
                    st.session_state.price_per_litre,
                )

                refined_input = create_refined_prompt(
                    start_prompt,
                    constitution,
                    st.session_state.llama_pipeline,
                    st.session_state.data,
                    st.session_state.max_budget,
                    st.session_state.predictions,
                    st.session_state.price_per_litre,
                )

                st.session_state.chat_history.append({"role": "assistant", "content": refined_input})
                st.session_state.local_chat.append({"role": "assistant", "content": refined_input})
                st.markdown(refined_input)

        # Mark the response as generated
        st.session_state.response_generated = True

        # Reset waiting_for_response to False to allow new input
        st.session_state.waiting_for_response = False

    # Chat input box (disabled if waiting for a response)
    if not st.session_state.waiting_for_response:
        user_input = st.chat_input("Ask something to AquaQuest AI...")
        if user_input:
            st.session_state.waiting_for_response = True
            # Add user message to chat history
            st.session_state.local_chat.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("AquaQuest AI is thinking..."):
                    # Simulate a response (replace with your model's output)
                    model_response, _ = conversation(
                        st.session_state.chat_history,
                        user_input,
                        st.session_state.llama_pipeline,
                        st.session_state.data,
                        st.session_state.max_budget,
                        st.session_state.predictions,
                        st.session_state.price_per_litre,
                    )
                    st.markdown(model_response)
                    st.session_state.local_chat.append({"role": "assistant", "content": model_response})

            # Reset waiting_for_response to False to allow new input
            st.session_state.waiting_for_response = False

# Main Function
def main():
    if st.session_state.page == 1:
        model_selection_page()
    elif st.session_state.page == 2:
        data_introduction_page()
    else:
        chatbot_page()

if __name__ == "__main__":
    main()