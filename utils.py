import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np

import matplotlib.pyplot as plt

import json

def plot_predictions(numerical_values, predictions):
    """
    Plots the actual numerical values as a line and the predictions as crosses at the next step (N+1).
    The y-axis starts at 0, and a margin is added above the plot.

    Parameters:
    - numerical_values: The actual numerical values (array-like).
    - predictions: The predicted values (array-like).
    """
    # Ensure numerical_values and predictions are 1D arrays for plotting
    numerical_values = numerical_values.flatten()
    predictions = predictions.flatten()

    # Create the x-axis for actual values (0 to N-1)
    x_actual = range(len(numerical_values))

    # Create the x-axis for predictions (N to N+len(predictions)-1)
    x_predictions = range(len(numerical_values), len(numerical_values) + len(predictions))

    # Determine the y-axis limits
    y_min = 0  # Start y-axis at 0
    y_max = max(max(numerical_values), max(predictions)) * 2  # Add 10% margin above the max value

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_actual, numerical_values, label="Actual Values", color="blue", linestyle="-", marker="o")
    plt.scatter(x_predictions, predictions, label="Predictions", color="red", marker="x", s=100)

    # Set y-axis limits
    plt.ylim(y_min, y_max)

    # Add labels, title, and legend
    plt.title("Actual Values vs Predictions", fontsize=16)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Show the plot
    plt.show()

def accumulate_water(df, accumulation_days):
    """
    Accumulates water values over a specified number of days (accumulation_days).

    Parameters:
        df (pd.DataFrame): The input DataFrame with a date index and water values as columns.
        accumulation_days (int): The number of days to accumulate water values.

    Returns:
        pd.DataFrame: A new DataFrame with accumulated values, dropping incomplete intervals.
    """
    # Ensure the DataFrame index is reset to integers for grouping
    df = df.reset_index(drop=True)

    # Calculate the number of complete intervals
    num_rows = len(df)
    complete_intervals = num_rows // accumulation_days

    # Drop rows that don't fit into complete intervals
    df = df.iloc[:complete_intervals * accumulation_days]

    # Group rows by accumulation intervals and sum the values
    accumulated_df = df.groupby(df.index // accumulation_days).sum()

    return accumulated_df


import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler



def predict_next_step(model, data):
    """
    Predicts the next value in a time series using a trained model.

    Parameters:
    - model: The trained TensorFlow/Keras model.
    - data: A pandas DataFrame containing the time series data. The last 12 rows are used for prediction.

    Returns:
    - prediction: The predicted next value (inverse-transformed).
    - numerical_values: The original numerical values (inverse-transformed).
    """
    # Extract the last 12 rows of data
    last_12_days = data.tail(12)

    # Extract the numerical values from the second column
    numerical_values = last_12_days.iloc[:, 1].values

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Reshape and scale the numerical values
    numerical_values = numerical_values.reshape(-1, 1)
    numerical_values_scaled = scaler.fit_transform(numerical_values)

    # Convert the scaled values to a TensorFlow tensor
    tensor = tf.convert_to_tensor(numerical_values_scaled, dtype=tf.float32)
    tensor = tf.reshape(tensor, (1, 12, 1))  # Add batch and feature dimensions

    # Make the prediction
    prediction_scaled = model.predict(tensor)

    # Inverse transform the prediction and numerical values
    prediction = scaler.inverse_transform(prediction_scaled)
    numerical_values = scaler.inverse_transform(numerical_values_scaled)

    return prediction.flatten(), numerical_values.flatten()

def plot_time_series(census_section, df):
    # Filter the DataFrame for the specified census section
    filtered_df = df[df['Secció censal/Sección censal/Census section'] == census_section].copy()

    # Convert the 'Data/Fecha/Date' column to datetime
    filtered_df['Data/Fecha/Date'] = pd.to_datetime(filtered_df['Data/Fecha/Date'])

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Data/Fecha/Date'],
        y=filtered_df['Average Consumption per Meter'],
        mode='lines+markers',
        name='Average Consumption'
    ))

    # Update layout to format the x-axis to show quarters
    fig.update_layout(
        title=f'Time Series of Average Consumption for Census Section {census_section}',
        xaxis_title='Date',
        yaxis_title='Average Consumption per Meter',
        xaxis=dict(
            tickformat='%Y-Q%q',  # Format to show year and quarter
            dtick="M3"  # Set ticks to every 3 months (quarterly)
        ),
        template='plotly_white'
    )

    # Show the plot
    fig.show()


def plot_cluster_statistics(cluster_ids, data, n = 5):
    """
    Plots the mean and standard deviation for a given cluster of customer IDs,
    and a sample of n time series from the cluster data.

    Parameters:
    - cluster_ids: List of customer IDs in the cluster.
    - data: DataFrame where columns are customer IDs and rows are time series data.
    - n: Number of samples to plot from the cluster data.

    Returns:
    - A Plotly figure showing the mean and standard deviation for the cluster on the left,
      and a sample of n time series from the cluster on the right.
    """
    # Extract the data for the given cluster
    cluster_data = data[cluster_ids]

    # Calculate the mean and standard deviation across the time series
    cluster_mean = cluster_data.mean(axis=1)
    cluster_std = cluster_data.std(axis=1)

    # Create a subplot figure with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cluster Statistics", "Sample Time Series"))

    # Add the mean line to the first subplot
    fig.add_trace(go.Scatter(
        x=cluster_data.index,
        y=cluster_mean,
        mode='lines',
        name='Mean',
        line=dict(color='blue')
    ), row=1, col=1)

    # Add the mean + std line to the first subplot
    fig.add_trace(go.Scatter(
        x=cluster_data.index,
        y=cluster_mean + cluster_std,
        mode='lines',
        name='Mean + Std',
        line=dict(color='green')
    ), row=1, col=1)

    # Add the mean - std line to the first subplot
    fig.add_trace(go.Scatter(
        x=cluster_data.index,
        y=cluster_mean - cluster_std,
        mode='lines',
        name='Mean - Std',
        line=dict(color='red')
    ), row=1, col=1)

    # Sample n time series from the cluster data
    sample_ids = np.random.choice(cluster_ids, min(n, len(cluster_ids)), replace=False)
    sample_data = cluster_data[sample_ids]

    # Add each sampled time series to the second subplot
    for sample_id in sample_ids:
        fig.add_trace(go.Scatter(
            x=sample_data.index,
            y=sample_data[sample_id],
            mode='lines',
            name=f'Sample {sample_id}'
        ), row=1, col=2)

    # Update layout
    fig.update_layout(
        title='Cluster Statistics and Sample Time Series',
        xaxis_title='Date',
        yaxis_title='Value',
        legend_title='Statistics',
        showlegend=False
    )

    # Show the plot
    fig.show()







start_system_prompt = """You are an advanced AI assistant specializing in helping users to analyse their water consumption data and achieve their target water budget. Your primary role is to analyze the provided data and provide accurate, evidence-driven recommendations or some encouragement that are actionable and easy to follow. You must interpret the given time series data and predicted consumption to offer insights that help users make informed decisions about their water usage. Your responses should focus on providing clear, helpful advice to achieve the user's goals while considering their budget and sustainability.

When providing recommendations, ensure they are based on the given data and are tailored to the user's specific context. Offer practical suggestions for reducing water consumption, optimizing usage, and staying within the given budget. Your tone should always be kind, empathetic, and supportive, fostering a friendly and constructive conversation with the user.

The INPUT will be provided in the following format:
{
    "person_prompt": "<The answer provided by the person>",
    "guideline": "<The guideline or constitution paragraph to critique the answer>",
    "time_series_data": [<Daily water consumption for the last 150 days>],
    "predicted_accumulation": {
        "1_day": <Predicted water consumption in Litres for the next 1 day>,
        "15_days": <Predicted water consumption in Litres for the next 15 days>,
        "30_days": <Predicted water consumption in Litres for the next 30 days>
    },
    "user_budget": <User's budget for the next 30 DAYS in Euros>,
    "predicted_price": <Predicted total cost of water consumption for the next 30 DAYS in Euros>
}

Your OUTPUT must follow this format:
"<Provide accurate, actionable recommendations based on the given data to help the user reduce water consumption, stay within budget, and achieve their goals. Start with a friendly introduction and provide insights into the data. End with a supportive closing statement.>"

IMPORTANT: Keep the output format exactly as specified, and do not deviate from it. Ensure the recommendations are realistic, actionable, and align with the user’s budget and water conservation goals. Avoid adding any additional words or explanations outside the specified structure.

Start the recommendation with "Hello, I am AquaQuest, your water consumption assistant. Based on the given data, " and finish with "If you have further questions or need specific advice, feel free to ask me anytime!". Use paragraphs and spaces for better comprehension, and write using a kind and supportive tone, addressing the user directly in the second person."""


conversation_system_prompt = """You are an advanced AI assistant specializing in helping users to analyse their water consumption data and achieve their target water budget. Your primary role is to analyze the provided data and provide accurate, evidence-driven recommendations or some encouragement that are actionable and easy to follow. You must interpret the given time series data and predicted consumption to offer insights that help users make informed decisions about their water usage. Your responses should focus on providing clear, helpful advice to achieve the user's goals while considering their budget and sustainability.

When providing recommendations, ensure they are based on the given data and are tailored to the user's specific context. Offer practical suggestions for reducing water consumption, optimizing usage, and staying within the given budget. Your tone should always be kind, empathetic, and supportive, fostering a friendly and constructive conversation with the user.

Answer the user's questions and provide them with the necessary information to help them make informed decisions about their water consumption, adapting the suggetsiong to the user's requirements and budget.

The INPUT will be provided in the following format:
{
    "person_prompt": "<The answer provided by the person>",
    "time_series_data": [<Daily water consumption for the last 150 days>],
    "predicted_accumulation": {
        "1_day": <Predicted water consumption in Litres for the next 1 day>,
        "15_days": <Predicted water consumption in Litres for the next 15 days>,
        "30_days": <Predicted water consumption in Litres for the next 30 days>
    },
    "user_budget": <User's budget for the next 30 DAYS in Euros>,
    "predicted_price": <Predicted total cost of water consumption for the next 30 DAYS in Euros>
}

Your OUTPUT must follow this format:
"<Your new reocmmendation based on the person_prompt>"

IMPORTANT: Keep the output format exactly as specified, and do not deviate from it. Ensure the recommendations are realistic, actionable, and align with the user’s budget and water conservation goals. Avoid adding any additional words or explanations outside the specified structure.

Continue the given conversation answering the questions of the user. Use paragraphs and spaces for better comprehension, and write using a kind and supportive tone, addressing the user directly in the second person."""




scaffolding_system_prompt = """You are an advanced AI assistant specializing in critically evaluating and improving other assistants responses based on a given set of guidelines or constitution. Your primary role is to analyze the provided answer in detail and provide constructive, evidence-driven critiques. This involves identifying specific issues such as logical flaws, inaccuracies, lack of alignment with the guideline, unclear language, or insufficient detail. You must clearly explain why these issues exist and how they detract from the quality of the prompt or answer.

In addition to critiquing, you are responsible for creating an improved version of the answer. This improved version must resolve all identified issues while maintaining full alignment with the provided guideline. The revised version should be clear, well-structured, and capable of achieving the intended objectives outlined in the guideline. Your improvements should focus on enhancing clarity, precision, and adherence to the specified requirements, ensuring the result is both actionable and effective.

When conducting critiques and providing improvements, it is essential to maintain a professional, respectful, and constructive tone. Your feedback must be clear, concise, and easy to understand so that it can be acted upon effectively. All critiques and improvements must be aligned with the principles of the provided guideline or constitution and must be driven by evidence and reasoning.

The INPUT will be provided in the following format:
{
    "person_prompt": "<The answer provided by the person>",
    "guideline": "<The guideline or constitution paragraph to critique the answer>",
    "time_series_data": [<Daily water consumption for the last 150 days>],
    "predicted_accumulation": {
        "1_day": <Predicted water consumption in Litres for the next 1 day>,
        "15_days": <Predicted water consumption in Litres for the next 15 days>,
        "30_days": <Predicted water consumption in Litres for the next 30 days>
    },
    "user_budget": <User's budget for the next 30 DAYS in Euros>,
    "predicted_price": <Predicted total cost of water consumption for the next 30 DAYS in Euros>
}

Your OUTPUT must follow this format:
"<An improved version of the given person's prompt after conducting a deep critique based on the provided guideline, keeping it concise>"


IMPORTANT: Keep the output format exactly as specified, and do not deviate from it. Ensure the improved version fully resolves all identified issues and strictly adheres to the requirements of the guideline. Avoid adding any additional words or explanations outside the specified structure.

Start the improved_answer answer with "Hello, I am AquaQuest your water consumtion assistant. Based on the given data, " and finish ecouraging the user to ask you more questions. Use paragraphs, spaces for a better comprehention nad write the imporved prompt using second person intead of mentioning the user.
"""


def generate_input_prompt(person_answer, guideline, time_series_data, predicted_accumulation, user_budget, predicted_price):
    """
    Generates the input JSON prompt for the AI system based on the given parameters.

    Parameters:
        person_answer (str): The answer provided by the person.
        guideline (str): The guideline or constitution paragraph to critique the answer.
        time_series_data (list): A list of daily water consumption values for the last 150 days.
        predicted_accumulation (dict): A dictionary containing predicted water consumption for the next 1 day, 15 days, and 30 days.
        user_budget (float): The user's budget for water consumption over the next 30 days.
        predicted_price (float): The predicted total cost of water consumption for the next 30 days.

    Returns:
        str: A JSON string formatted as the input prompt.
    """
    # Create the input JSON structure
    input_json = {
        "person_answer": person_answer,
        "guideline": guideline,
        "time_series_data": time_series_data,
        "predicted_accumulation": predicted_accumulation,
        "user_budget": user_budget,
        "predicted_price": predicted_price
    }

    # Return the JSON as a formatted string
    return json.dumps(input_json, indent=4)


def create_predictions(model, dataset):
    one_day_prediction, numerical_values = predict_next_step(model, dataset)
    accumulated_data = accumulate_water(dataset, 15)
    fifteen_day_prediction, numerical_values = predict_next_step(model, accumulated_data)
    accumulated_data = accumulate_water(dataset, 30)
    thirty_day_prediction, numerical_values = predict_next_step(model, accumulated_data)
    return { "1_day": str(one_day_prediction[0]),  "15_days": str(fifteen_day_prediction[0]), "30_days": str(thirty_day_prediction[0])}

def get_last_days(dataset):
    last_days = dataset.tail(150)
    return last_days.iloc[:, 1].values.tolist()
        


def create_start_prompt(pipe, data, user_budget, predictions, price_per_litre):
    last_days = get_last_days(data)
    predictions = predictions
    person_prompt = "Given my water usage and my budget could you kindly provide me some recmmendations based on the data provieded?"
    input_json = {
        "person_answer": person_prompt,
        "time_series_data": last_days,
        "predicted_accumulation": predictions,
        "user_budget": str(round(user_budget,2)),
        "predicted_price": str(round(float(predictions["30_days"]) * price_per_litre,2))
    }

    input_json_str = json.dumps(input_json)


    chat_history = [
        {"role": "system", "content": start_system_prompt},
    ]

    # Append user message to chat history
    chat_history.append({"role": "user", "content": input_json_str})

    # Generate response
    outputs = pipe(
        chat_history,
        max_new_tokens=1000,
    )

    # Extract the model's response
    model_response = outputs[0]["generated_text"][-1]["content"]

    # Append model response to chat history
    chat_history.append({"role": "assistant", "content": model_response})

    return model_response, input_json_str



constitution = [
   "Provide a clear and detailed explanation of the data to help the user fully understand it, ensuring they can accurately interpret their water usage and determine whether they are staying within their budget or exceeding it comparing the user_budget and the predicted_budget. If the data indicates they are exceeding their budget, offer specific, actionable recommendations to reduce water consumption, such as adjusting daily habits or implementing water-saving technologies. Conversely, if they are staying within their budget, congratulate them on their success and provide general tips to further optimize their water usage, like monitoring for leaks or using water-efficient appliances. All recommendations should be directly tied to the provided data, logically derived, and explicitly supported by the evidence to ensure accuracy, relevance, and practical value."
]

def create_refined_prompt(input, constitution, pipe, data, user_budget, predictions, price_per_litre):

    last_days = get_last_days(data)
    predictions = predictions

    user_budget = str(round(user_budget,2)),
    predicted_price = str(round(float(predictions["30_days"]) * price_per_litre,2))

    model_response = input
 

    for guideline in constitution:

        input_json_str = generate_input_prompt(input, guideline, last_days, predictions, user_budget, predicted_price)

        chat_history = [
            {"role": "system", "content": scaffolding_system_prompt},
        ]

        # Append user message to chat history
        chat_history.append({"role": "user", "content": input_json_str})

        # Generate response
        outputs = pipe(
            chat_history,
            max_new_tokens=2000,
        )

        # Extract the model's response
        model_response = outputs[0]["generated_text"][-1]["content"]

        # Append model response to chat history
        chat_history.append({"role": "assistant", "content": model_response})


    return model_response


def conversation(chat_history,  person_prompt, pipe, data, user_budget, predictions, price_per_litre):
    last_days = get_last_days(data)
    predictions = predictions
    input_json = {
        "person_answer": person_prompt,
        "time_series_data": last_days,
        "predicted_accumulation": predictions,
        "user_budget": str(round(user_budget,2)),
        "predicted_price": str(round(float(predictions["30_days"]) * price_per_litre,2))
    }

    input_json_str = json.dumps(input_json)

    # Append user message to chat history
    chat_history.append({"role": "user", "content": input_json_str})

    # Generate response
    outputs = pipe(
        chat_history,
        max_new_tokens=2000,
    )

    # Extract the model's response
    model_response = outputs[0]["generated_text"][-1]["content"]

    # Append model response to chat history
    chat_history.append({"role": "assistant", "content": model_response})

    return model_response, chat_history