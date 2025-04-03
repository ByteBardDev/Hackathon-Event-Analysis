import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from faker import Faker
import io
import random
from datetime import datetime, timedelta
import altair as alt
from wordcloud import WordCloud
import base64
from io import BytesIO
import streamlit as st
from PIL import Image
import os
from io import BytesIO

fake = Faker('en_IN')


# Set page configuration
st.set_page_config(
    page_title="Hackathon Event Analysis",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the look and feel
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to generate synthetic dataset
@st.cache_data
def generate_dataset():
    # Constants
    num_participants = 350
    used_names = set()
    names = []

    while len(names) < num_participants:
      name = fake.first_name() + ' ' + fake.last_name()

      # Ensure unique names
      if name not in used_names:
        used_names.add(name)
        names.append(name)

    domains = ["Web Development", "Machine Learning", "Blockchain", "IoT", "Mobile App Development"]
    colleges = ["MIT Tech Institute", "Stanford College", "Harvard University", "IIT Mumbai",
                "Oxford University", "Cambridge Tech", "Yale Institute", "Princeton College",
                "Columbia University", "Berkeley Institute"]
    states = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Telangana",
              "Gujarat", "West Bengal", "Uttar Pradesh", "Rajasthan", "Kerala"]

    # Feedback templates
    positive_feedback = [
        "Really enjoyed the hackathon. Great learning experience!",
        "Excellent organization and mentorship throughout the event.",
        "Challenging but rewarding experience. Would participate again!",
        "Amazing opportunity to network with industry professionals.",
        "Very well organized event with good infrastructure."
    ]

    neutral_feedback = [
        "The event was okay. Some improvements needed in organization.",
        "Good concept but execution could be better.",
        "Average experience overall. Expected better mentorship.",
        "Decent hackathon but lacked proper technical support.",
        "Satisfactory experience with room for improvement."
    ]

    negative_feedback = [
        "Poor internet connectivity throughout the event.",
        "Not enough mentors available for guidance.",
        "Too much time pressure, couldn't complete the project.",
        "Lack of proper facilities and refreshments.",
        "Judges didn't provide constructive feedback."
    ]

    # Generate data
    start_date = datetime(2023, 9, 15)

    data = {
        'Participant_ID': [f'P{i+1:03d}' for i in range(num_participants)],
        'Name': names,
        'Age': np.random.randint(18, 35, num_participants),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], num_participants, p=[0.55, 0.40, 0.05]),
        'College': np.random.choice(colleges, num_participants),
        'State': np.random.choice(states, num_participants),
        'Domain': np.random.choice(domains, num_participants),
        'Day': np.random.choice([1, 2, 3], num_participants),  # Day 1, 2, or 3
        'Team_Size': np.random.choice([1, 2, 3, 4, 5], num_participants),
        'Completed_Project': np.random.choice([True, False], num_participants, p=[0.85, 0.15])
    }

    # Generate feedback based on domain and with varying sentiment
    feedback = []
    for domain in data['Domain']:
        sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=[0.6, 0.3, 0.1])
        if sentiment == 'positive':
            template = random.choice(positive_feedback)
        elif sentiment == 'neutral':
            template = random.choice(neutral_feedback)
        else:
            template = random.choice(negative_feedback)

        # Add domain-specific keywords to make feedback more relevant
        domain_keywords = {
            "Web Development": ["HTML", "CSS", "JavaScript", "responsive", "frontend", "backend"],
            "Machine Learning": ["algorithm", "model", "dataset", "prediction", "accuracy", "training"],
            "Blockchain": ["smart contract", "decentralized", "crypto", "ledger", "tokens", "mining"],
            "IoT": ["sensors", "devices", "connectivity", "automation", "monitoring", "embedded"],
            "Mobile App Development": ["UI/UX", "native", "cross-platform", "Flutter", "React Native", "mobile"]
        }

        # Add 1-2 domain keywords to the feedback
        keyword_count = random.randint(1, 2)
        selected_keywords = random.sample(domain_keywords[domain], keyword_count)
        domain_specific_part = f" The {', '.join(selected_keywords)} aspects were particularly noteworthy."

        complete_feedback = template + domain_specific_part
        feedback.append(complete_feedback)

    data['Feedback'] = feedback

    return pd.DataFrame(data)

# Generate sample images for each domain and day
@st.cache_data
def generate_sample_images():
    domains = ["Web Development", "Machine Learning", "Blockchain", "IoT", "Mobile App Development"]
    days = [1, 2, 3]
    images = {}

    # Create colored rectangles with text as sample images
    for domain in domains:
        domain_images = {}
        for day in days:
            # Create a colored image
            if domain == "Web Development":
                color = (41, 128, 185)  # Blue
            elif domain == "Machine Learning":
                color = (39, 174, 96)  # Green
            elif domain == "Blockchain":
                color = (142, 68, 173)  # Purple
            elif domain == "IoT":
                color = (230, 126, 34)  # Orange
            else:  # Mobile App Development
                color = (231, 76, 60)  # Red

            img = Image.new('RGB', (400, 300), color)

            # Save image to BytesIO object
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            domain_images[day] = img_byte_arr.getvalue()

        images[domain] = domain_images

    return images

# Function to apply custom filters to images
def apply_image_filter(image_bytes, filter_name):
    image = Image.open(BytesIO(image_bytes))

    if filter_name == "Grayscale":
        return np.array(image.convert('L'))
    elif filter_name == "Sepia":
        img_array = np.array(image.convert('RGB'))
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        sepia_img = img_array.dot(sepia_filter.T)
        sepia_img /= sepia_img.max()
        sepia_img *= 255
        return sepia_img.astype(np.uint8)
    elif filter_name == "Invert":
        return 255 - np.array(image)
    elif filter_name == "Blur":
        from scipy.ndimage import gaussian_filter
        img_array = np.array(image)
        return gaussian_filter(img_array, sigma=3)
    else:  # No filter
        return np.array(image)

# Function to create word clouds
def generate_word_cloud(text, colormap='viridis'):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        contour_width=3
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf

# Main application
def main():
    # Generate the dataset
    df = generate_dataset()
    sample_images = generate_sample_images()

    # Sidebar navigation
    st.sidebar.title("🧩 Hackathon Analytics")
    page = st.sidebar.radio("Welcome to Christ HackaVerse", [
        "🏠 Home",
        "📊 Dashboard",
        "💬 Feedback Analysis",
        "📸 Image Gallery"
    ])

    # Download dataset option in sidebar
    if st.sidebar.button("Download Dataset"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="hackathon_data.csv">Download CSV File</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    # Home page
    if page == "🏠 Home":
        st.title("🧩 Hackathon Event Analysis")
        st.markdown("### Welcome to the Hackathon Event Analytics Dashboard")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            This interactive dashboard provides comprehensive insights into our recent hackathon event.
            Explore participation trends, feedback analysis, and an image gallery from the event.

            ### Features:
            - **Dashboard**: Visualize participation data across domains, colleges, and states
            - **Feedback Analysis**: Understand participant sentiment with word clouds and text analysis
            - **Image Gallery**: Browse event photos with custom image processing

            Use the navigation panel on the left to explore different sections of the dashboard.
            """)

            # Key metrics
            st.subheader("Key Metrics")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("Total Participants", len(df))
            with metrics_col2:
                st.metric("Domains", len(df['Domain'].unique()))
            with metrics_col3:
                st.metric("Event Days", df['Day'].nunique())
            with metrics_col4:
                completion_rate = round(df['Completed_Project'].mean() * 100, 1)
                st.metric("Project Completion Rate", f"{completion_rate}%")

        with col2:
            # Quick domain distribution chart
            st.subheader("Domain Distribution")
            domain_counts = df['Domain'].value_counts().reset_index()
            domain_counts.columns = ['Domain', 'Count']

            chart = alt.Chart(domain_counts).mark_bar().encode(
                y=alt.Y('Domain:N', sort='-x', title=None),
                x=alt.X('Count:Q', title='Number of Participants'),
                color=alt.Color('Domain:N', legend=None),
                tooltip=['Domain', 'Count']
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

            # Dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(5), use_container_width=True)

    # Dashboard page
    elif page == "📊 Dashboard":
        st.title("📊 Hackathon Participation Dashboard")

        # Filters in sidebar
        st.sidebar.header("Filters")
        selected_domains = st.sidebar.multiselect(
            "Select Domains",
            options=sorted(df['Domain'].unique()),
            default=sorted(df['Domain'].unique())
        )

        selected_states = st.sidebar.multiselect(
            "Select States",
            options=sorted(df['State'].unique()),
            default=sorted(df['State'].unique())[:3]  # Default to first 3 states
        )

        selected_colleges = st.sidebar.multiselect(
            "Select Colleges",
            options=sorted(df['College'].unique()),
            default=sorted(df['College'].unique())[:3]  # Default to first 3 colleges
        )

        # Apply filters
        filtered_df = df[
            df['Domain'].isin(selected_domains) &
            df['State'].isin(selected_states) &
            df['College'].isin(selected_colleges)
        ]

        # Display filtered data count
        st.markdown(f"### Showing data for {len(filtered_df)} participants")

        # 1. Domain-wise Distribution
        st.subheader("1. Domain-wise Participation")
        col1, col2 = st.columns([2, 1])

        with col1:
            domain_counts = filtered_df['Domain'].value_counts().reset_index()
            domain_counts.columns = ['Domain', 'Count']

            domain_chart = alt.Chart(domain_counts).mark_bar().encode(
                x=alt.X('Domain:N', title='Domain', sort='-y'),
                y=alt.Y('Count:Q', title='Number of Participants'),
                color=alt.Color('Domain:N', legend=None),
                tooltip=['Domain', 'Count']
            ).properties(height=300)

            st.altair_chart(domain_chart, use_container_width=True)

        with col2:
            # Pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(domain_counts['Count'], labels=domain_counts['Domain'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        # 2. Day-wise Participation
        st.subheader("2. Day-wise Participation")
        day_domain = pd.crosstab(filtered_df['Day'], filtered_df['Domain'])

        # Stacked bar chart for day-wise domain distribution
        day_domain_df = day_domain.reset_index().melt(id_vars=['Day'], var_name='Domain', value_name='Count')
        day_domain_df['Day'] = day_domain_df['Day'].astype(str)

        chart = alt.Chart(day_domain_df).mark_bar().encode(
            x=alt.X('Day:N', title='Day'),
            y=alt.Y('Count:Q', title='Number of Participants'),
            color=alt.Color('Domain:N', title='Domain'),
            tooltip=['Day', 'Domain', 'Count']
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)

        # 3. College-wise Distribution
        st.subheader("3. College-wise Participation")
        college_counts = filtered_df['College'].value_counts().reset_index()
        college_counts.columns = ['College', 'Count']
        college_counts = college_counts.sort_values('Count', ascending=False).head(10)

        chart = alt.Chart(college_counts).mark_bar().encode(
            y=alt.Y('College:N', sort='-x', title='College'),
            x=alt.X('Count:Q', title='Number of Participants'),
            color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['College', 'Count']
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

        # 4. State-wise Distribution
        st.subheader("4. State-wise Participation")
        col1, col2 = st.columns([3, 2])

        with col1:
            state_counts = filtered_df['State'].value_counts().reset_index()
            state_counts.columns = ['State', 'Count']

            chart = alt.Chart(state_counts).mark_bar().encode(
                x=alt.X('State:N', sort='-y', title='State'),
                y=alt.Y('Count:Q', title='Number of Participants'),
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='greens'), legend=None),
                tooltip=['State', 'Count']
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

        with col2:
            # Table with state data
            state_counts = state_counts.sort_values('Count', ascending=False)
            st.dataframe(state_counts, use_container_width=True)

        # 5. Gender Distribution by Domain
        st.subheader("5. Gender Distribution by Domain")
        gender_domain = pd.crosstab(filtered_df['Gender'], filtered_df['Domain'])
        gender_domain_df = gender_domain.reset_index().melt(id_vars=['Gender'], var_name='Domain', value_name='Count')

        chart = alt.Chart(gender_domain_df).mark_bar().encode(
            x=alt.X('Domain:N', title='Domain'),
            y=alt.Y('Count:Q', title='Number of Participants'),
            color=alt.Color('Gender:N', title='Gender'),
            tooltip=['Domain', 'Gender', 'Count']
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)

        # 6. Age Distribution
        st.subheader("6. Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df['Age'], kde=True, bins=15, ax=ax)
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    # Feedback Analysis page
    elif page == "💬 Feedback Analysis":
        st.title("💬 Participant Feedback Analysis")

        # Domain filter for feedback
        selected_domain = st.selectbox(
            "Select Domain for Feedback Analysis",
            options=sorted(df['Domain'].unique())
        )

        domain_df = df[df['Domain'] == selected_domain]

        # Combine all feedback for the selected domain
        all_feedback = " ".join(domain_df['Feedback'].tolist())

        # Generate word cloud
        st.subheader(f"Word Cloud for {selected_domain}")

        col1, col2 = st.columns([1, 3])

        with col1:
            colormap = st.selectbox(
                "Select Color Theme",
                options=['viridis', 'plasma', 'inferno', 'magma', 'Blues', 'Greens', 'Reds']
            )

        with col2:
            wordcloud_bytes = generate_word_cloud(all_feedback, colormap)
            st.image(wordcloud_bytes, use_column_width=True)

        # Sample feedback display
        st.subheader(f"Sample Feedback for {selected_domain}")
        sample_size = min(5, len(domain_df))
        sample_feedback = domain_df.sample(sample_size)

        for i, (_, row) in enumerate(sample_feedback.iterrows()):
            st.markdown(f"""
            **Participant {row['Participant_ID']}** (Day {row['Day']})
            > {row['Feedback']}
            """)
            if i < sample_size - 1:
                st.markdown("---")

        # Word frequency analysis
        st.subheader("Keyword Analysis")

        # Extract keywords based on domain
        domain_keywords = {
            "Web Development": ["HTML", "CSS", "JavaScript", "responsive", "frontend", "backend"],
            "Machine Learning": ["algorithm", "model", "dataset", "prediction", "accuracy", "training"],
            "Blockchain": ["smart contract", "decentralized", "crypto", "ledger", "tokens", "mining"],
            "IoT": ["sensors", "devices", "connectivity", "automation", "monitoring", "embedded"],
            "Mobile App Development": ["UI/UX", "native", "cross-platform", "Flutter", "React Native", "mobile"]
        }

        # Calculate frequency of each keyword
        keyword_freq = {}
        for keyword in domain_keywords[selected_domain]:
            keyword_freq[keyword] = all_feedback.lower().count(keyword.lower())

        # Create dataframe and chart
        keyword_df = pd.DataFrame({
            'Keyword': list(keyword_freq.keys()),
            'Frequency': list(keyword_freq.values())
        }).sort_values('Frequency', ascending=False)

        chart = alt.Chart(keyword_df).mark_bar().encode(
            x=alt.X('Frequency:Q', title='Frequency'),
            y=alt.Y('Keyword:N', sort='-x', title='Keyword'),
            color=alt.Color('Frequency:Q', scale=alt.Scale(scheme=colormap), legend=None),
            tooltip=['Keyword', 'Frequency']
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)

        # Compare domains (common positive/negative words)
        st.subheader("Domain Comparison")

        # Common sentiment words
        positive_words = ["enjoyed", "excellent", "great", "amazing", "good", "well"]
        negative_words = ["poor", "lack", "not enough", "couldn't", "better"]

        # Calculate sentiment scores by domain
        sentiment_data = []

        for domain in df['Domain'].unique():
            domain_text = " ".join(df[df['Domain'] == domain]['Feedback'].tolist()).lower()

            positive_score = sum(domain_text.count(word) for word in positive_words)
            negative_score = sum(domain_text.count(word) for word in negative_words)

            sentiment_data.append({
                'Domain': domain,
                'Positive': positive_score,
                'Negative': negative_score,
                'Net': positive_score - negative_score
            })

        sentiment_df = pd.DataFrame(sentiment_data)

        # Create chart
        sentiment_comp = sentiment_df.melt(
            id_vars=['Domain'],
            value_vars=['Positive', 'Negative'],
            var_name='Sentiment',
            value_name='Score'
        )

        chart = alt.Chart(sentiment_comp).mark_bar().encode(
            x=alt.X('Domain:N', title='Domain'),
            y=alt.Y('Score:Q', title='Sentiment Score'),
            color=alt.Color('Sentiment:N', scale=alt.Scale(domain=['Positive', 'Negative'],
                                                         range=['#2ecc71', '#e74c3c'])),
            tooltip=['Domain', 'Sentiment', 'Score']
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)

        # Net sentiment score
        chart = alt.Chart(sentiment_df).mark_bar().encode(
            x=alt.X('Domain:N', title='Domain'),
            y=alt.Y('Net:Q', title='Net Sentiment Score'),
            color=alt.condition(
                alt.datum.Net > 0,
                alt.value('#2ecc71'),  # green for positive
                alt.value('#e74c3c')   # red for negative
            ),
            tooltip=['Domain', 'Net']
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)

    # Image Gallery page
    elif page == "📸 Image Gallery":
        st.title("📸 Hackathon Image Gallery")

        # Domain and day selection
        col1, col2 = st.columns(2)

        with col1:
            selected_domain = st.selectbox(
                "Select Domain",
                options=sorted(df['Domain'].unique())
            )

        with col2:
            selected_day = st.radio(
                "Select Day",
                options=[1, 2, 3],
                horizontal=True
            )

        # Display image for selected domain and day
        if selected_domain and selected_day:
            st.subheader(f"{selected_domain} - Day {selected_day}")

            # Get image bytes
            image_bytes = sample_images[selected_domain][selected_day]

            # Image processing options
            filter_options = ["No Filter", "Grayscale", "Sepia", "Invert", "Blur"]
            selected_filter = st.selectbox("Apply Image Filter", filter_options)

            # Apply selected filter
            if selected_filter != "No Filter":
                processed_image = apply_image_filter(image_bytes, selected_filter)
                st.image(processed_image, caption=f"{selected_domain} - Day {selected_day} with {selected_filter} filter")
            else:
                st.image(image_bytes, caption=f"{selected_domain} - Day {selected_day}")

        # Image gallery for all days of selected domain
        st.subheader(f"All Days - {selected_domain}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(sample_images[selected_domain][1], caption=f"Day 1")

        with col2:
            st.image(sample_images[selected_domain][2], caption=f"Day 2")

        with col3:
            st.image(sample_images[selected_domain][3], caption=f"Day 3")

        # Custom image processing experiment
        st.subheader("Custom Image Processing")
        st.write("Upload your own image to apply filters:")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read image file
            image_bytes = uploaded_file.getvalue()

            col1, col2 = st.columns(2)

            with col1:
                st.image(image_bytes, caption="Original Image")

            with col2:
                filter_type = st.selectbox(
                    "Select Filter",
                    options=["Grayscale", "Sepia", "Invert", "Blur"]
                )

                processed = apply_image_filter(image_bytes, filter_type)
                st.image(processed, caption=f"With {filter_type} filter")

# Run the app
if __name__ == "__main__":
    main()
