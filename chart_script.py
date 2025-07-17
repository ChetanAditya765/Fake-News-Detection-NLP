import plotly.graph_objects as go
import plotly.express as px
import json

# Data for the system architecture
data = {
    "components": [
        {"layer": "Data Input", "items": ["News Articles", "Social Media", "Headlines", "Article Text"]},
        {"layer": "Data Preprocessing", "items": ["Text Clean", "Tokenization", "Stop Word Rm", "Lemmatization"]},
        {"layer": "Feature Extraction", "items": ["TF-IDF Vector", "Word2Vec Embed", "BERT Embed", "N-gram Feat"]},
        {"layer": "Model Training", "items": ["Logistic Reg", "Random Forest", "LSTM Network", "BERT Fine-tune"]},
        {"layer": "Explainable AI", "items": ["LIME Explain", "SHAP Values", "Attention Wt", "Feature Imp"]},
        {"layer": "Prediction", "items": ["Classification", "Confidence", "Explanation", "Reasoning"]},
        {"layer": "User Interface", "items": ["Web Interface", "Input Form", "Results Disp", "Visualization"]}
    ]
}

# Create the figure
fig = go.Figure()

# Define colors for each layer
colors = ['#1FB8CD', '#FFC185', '#ECEBD5', '#5D878F', '#D2BA4C', '#B4413C', '#964325']

# Define positions for each layer (y-coordinates)
layer_positions = {
    "Data Input": 6,
    "Data Preprocessing": 5,
    "Feature Extraction": 4,
    "Model Training": 3,
    "Explainable AI": 2,
    "Prediction": 1,
    "User Interface": 0
}

# Add layer boxes and component items
for i, component in enumerate(data["components"]):
    layer_name = component["layer"]
    items = component["items"]
    y_pos = layer_positions[layer_name]
    
    # Add main layer box
    fig.add_shape(
        type="rect",
        x0=0.1, y0=y_pos-0.4,
        x1=0.9, y1=y_pos+0.4,
        fillcolor=colors[i],
        opacity=0.8,
        line=dict(color="white", width=2)
    )
    
    # Add layer title
    fig.add_annotation(
        x=0.5, y=y_pos+0.25,
        text=f"<b>{layer_name}</b>",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Add component items in a grid
    x_positions = [0.2, 0.4, 0.6, 0.8]
    for j, item in enumerate(items):
        if j < 4:  # Limit to 4 items per layer
            fig.add_annotation(
                x=x_positions[j], y=y_pos-0.1,
                text=item,
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                borderpad=2
            )

# Add arrows between layers
for i in range(len(data["components"]) - 1):
    current_layer = data["components"][i]["layer"]
    next_layer = data["components"][i + 1]["layer"]
    
    y_start = layer_positions[current_layer]
    y_end = layer_positions[next_layer]
    
    # Add arrow
    fig.add_annotation(
        x=0.5, y=y_start-0.45,
        ax=0.5, ay=y_end+0.45,
        xref="x", yref="y",
        axref="x", ayref="y",
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor="gray"
    )

# Add side labels for flow direction
fig.add_annotation(
    x=1.0, y=3,
    text="Data Flow",
    showarrow=False,
    font=dict(size=12, color="gray"),
    textangle=-90
)

# Update layout
fig.update_layout(
    title="Fake News Detection System",
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, 1.2]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.5, 6.5]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False
)

# Save the chart
fig.write_image("fake_news_architecture.png")