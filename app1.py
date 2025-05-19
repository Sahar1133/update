# ====================== IMPORTS ======================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random
from PIL import Image
import base64

# ====================== BACKGROUND IMAGE ======================
def add_bg_from_local(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ====================== STYLING & SETUP ======================
def apply_custom_css():
    """Applies custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
    /* Main content area with glassmorphism effect */
    .main .block-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Header styling with gradient text */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1.5rem;
        font-size: 2.5rem;
    }
    
    /* Card styling with subtle animation */
    .stCard {
        background: white;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        padding: 1.75rem;
        margin-bottom: 1.75rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.12);
    }
    
    /* Button styling with pulse animation */
    .stButton>button {
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 14px 32px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(107, 115, 255, 0.4);
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(107, 115, 255, 0.6);
    }
    .stButton>button:after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0) 100%);
        opacity: 0;
        transition: all 0.3s ease;
    }
    .stButton>button:hover:after {
        opacity: 1;
    }
    
    /* Radio buttons with modern styling */
    .stRadio > div {
        flex-direction: column;
        gap: 12px;
    }
    .stRadio > div > label {
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(5px);
        padding: 18px;
        border-radius: 12px;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stRadio > div > label:hover {
        background: rgba(255,255,255,0.9);
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .stRadio > div > label[data-baseweb="radio"]:first-child {
        margin-top: 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 12px;
        transition: all 0.3s;
        background: rgba(255,255,255,0.7);
        margin-right: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        color: #6B73FF;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .stTabs [aria-selected="false"] {
        background: rgba(255,255,255,0.5);
        color: #4a5568;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: white;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(0,0,0,0.2);
        color: white;
        border-color: rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(0,0,0,0.3);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.5);
    }
    
    /* Success message styling */
    .stAlert .st-ae {
        background: rgba(102, 187, 106, 0.9) !important;
        backdrop-filter: blur(5px);
        border-radius: 12px;
        border: none;
    }
    
    /* Expander styling */
    .stExpander {
        background: rgba(255,255,255,0.8);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stExpander > summary {
        font-weight: 600;
        padding: 1rem 1.5rem;
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== DATA LOADING & PREPROCESSING ======================
@st.cache_data
def load_data():
    career_options = [
        'Software Developer', 'Data Scientist', 'AI Engineer', 
        'Cybersecurity Specialist', 'Cloud Architect',
        'Marketing Manager', 'Financial Analyst', 'HR Manager',
        'Entrepreneur', 'Investment Banker',
        'Graphic Designer', 'Video Editor', 'Music Producer',
        'Creative Writer', 'Art Director',
        'Mechanical Engineer', 'Electrical Engineer', 
        'Civil Engineer', 'Robotics Engineer',
        'Doctor', 'Nurse', 'Psychologist', 
        'Physical Therapist', 'Medical Researcher',
        'Biotechnologist', 'Research Scientist', 
        'Environmental Scientist', 'Physicist',
        'Teacher', 'Professor', 'Educational Consultant',
        'Curriculum Developer',
        'Lawyer', 'Judge', 'Legal Consultant',
        'UX Designer', 'Product Manager',
        'Journalist', 'Public Relations Specialist',
        'Architect', 'Urban Planner',
        'Chef', 'Event Planner', 'Fashion Designer'
    ]
    
    try:
        data = pd.read_excel("new.xlsx")
        if len(data['Predicted_Career_Field'].unique()) < 20:
            data['Predicted_Career_Field'] = np.random.choice(career_options, size=len(data))
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Using demo data.")
        data = pd.DataFrame({
            'Interest': np.random.choice(['Technology', 'Business', 'Arts', 'Engineering', 'Medical', 'Science', 'Education', 'Law'], 200),
            'Work_Style': np.random.choice(['Independent', 'Collaborative', 'Flexible'], 200),
            'Strengths': np.random.choice(['Analytical', 'Creative', 'Strategic', 'Practical'], 200),
            'Communication_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Leadership_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Teamwork_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'GPA': np.round(np.random.uniform(2.0, 4.0, 200), 1),
            'Years_of_Experience': np.random.randint(0, 20, 200),
            'Predicted_Career_Field': np.random.choice(career_options, 200)
        })
    
    if 'GPA' in data.columns:
        data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
        data['GPA'].fillna(data['GPA'].median(), inplace=True)
    
    return data

# ====================== MODEL TRAINING ======================
def preprocess_data(data):
    le = LabelEncoder()
    object_cols = [col for col in data.select_dtypes(include=['object']).columns 
                  if col in data.columns]
    for col in object_cols:
        if col != 'Predicted_Career_Field':
            data[col] = le.fit_transform(data[col].astype(str))
    if 'Predicted_Career_Field' in data.columns:
        data['Predicted_Career_Field'] = le.fit_transform(data['Predicted_Career_Field'])
    return data, le

def train_model(data):
    if 'Predicted_Career_Field' not in data.columns:
        st.error("Target column not found in data")
        return None, 0
    
    X = data.drop('Predicted_Career_Field', axis=1)
    y = data['Predicted_Career_Field']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# ====================== QUESTIONNAIRE ======================
def get_all_questions():
    """Returns a pool of 30 questions"""
    return [
        # Interest Questions (1-8)
        {
            "question": "1. Which of these activities excites you most?",
            "options": [
                {"text": "Coding or working with technology", "value": "Technology"},
                {"text": "Analyzing market trends", "value": "Business"},
                {"text": "Creating art or designs", "value": "Arts"},
                {"text": "Building or fixing mechanical things", "value": "Engineering"},
                {"text": "Helping people with health issues", "value": "Medical"},
                {"text": "Conducting experiments", "value": "Science"},
                {"text": "Teaching others", "value": "Education"},
                {"text": "Debating or solving legal problems", "value": "Law"}
            ],
            "feature": "Interest"
        },
        # ... (keep all the questions from your original code)
    ]

def get_randomized_questions():
    """Selects 10 random questions from the pool of 30."""
    all_questions = get_all_questions()
    features = list(set(q['feature'] for q in all_questions))
    selected = []

    for feature in features:
        feature_questions = [q for q in all_questions if q['feature'] == feature]
        if feature_questions:
            selected.append(random.choice(feature_questions))

    remaining = [q for q in all_questions if q not in selected]
    needed = 10 - len(selected)

    if needed > 0 and remaining:
        selected.extend(random.sample(remaining, min(needed, len(remaining))))

    random.shuffle(selected)
    return selected

direct_input_features = {
    "GPA": {
        "question": "What is your approximate GPA (0.0-4.0)?",
        "type": "number", 
        "min": 0.0, 
        "max": 4.0, 
        "step": 0.1, 
        "default": 3.0
    },
    "Years_of_Experience": {
        "question": "Years of professional experience (if any):",
        "type": "number", 
        "min": 0, 
        "max": 50, 
        "step": 1, 
        "default": 0
    }
}

# ====================== MAIN APP ======================
def main():
    # Set background and styling
    add_bg_from_local("background.jpg")  # Using online image instead
    apply_custom_css()
    
    # Initialize session state
    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = {}
    if 'questions' not in st.session_state:
        st.session_state.questions = get_randomized_questions()

    # Load data and train model
    data = load_data()
    processed_data, target_le = preprocess_data(data.copy())
    model, accuracy = train_model(processed_data)

    # Page header with animated gradient
    st.markdown("""
    <h1 style="text-align: center; animation: gradient 5s ease infinite; background-size: 200% 200%;">
    🧭 AI Career Navigator
    </h1>
    <p style="text-align: center; font-size: 1.1rem; color: #4a5568;">
    Discover your ideal career path based on your unique personality traits and skills
    </p>
    """, unsafe_allow_html=True)

    # Sidebar with animated profile card
    with st.sidebar:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 16px; 
                    border: 1px solid rgba(255,255,255,0.2); margin-bottom: 2rem;">
            <h3 style="color: white; text-align: center;">🔍 About This Tool</h3>
            <p style="color: rgba(255,255,255,0.8);">
            This AI-powered assessment analyzes your personality traits, skills, and preferences 
            to match you with suitable career options from our database of {len(data)} career paths.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 16px; 
                    border: 1px solid rgba(255,255,255,0.2);">
            <h3 style="color: white; text-align: center;">📊 Model Accuracy</h3>
            <div style="background: rgba(0,0,0,0.2); border-radius: 12px; padding: 1rem; text-align: center;">
                <h1 style="color: #4fd1c5; margin: 0;">{accuracy*100:.1f}%</h1>
                <p style="color: rgba(255,255,255,0.7); margin: 0;">Prediction Accuracy</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2 = st.tabs(["✨ Take Assessment", "📊 Career Insights"])

    with tab1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.8); border-radius: 16px; padding: 1.5rem; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 2rem;">
            <h2 style="color: #2d3748;">Career Compatibility Assessment</h2>
            <p style="color: #4a5568;">
            Answer these 10 questions honestly to discover careers that best fit your personality and skills.
            The more accurate your responses, the better our AI can match you with suitable options.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Background information section with card styling
        with st.expander("📝 Your Background Information", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                st.session_state.user_responses["GPA"] = st.number_input(
                    "What is your approximate GPA (0.0-4.0)?",
                    min_value=0.0,
                    max_value=4.0,
                    value=3.0,
                    step=0.1,
                    key="num_GPA"
                )
            with cols[1]:
                st.session_state.user_responses["Years_of_Experience"] = st.number_input(
                    "Years of professional experience (if any):",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1,
                    key="num_Years_of_Experience"
                )

        # Personality questions with animated cards
        st.markdown("""
        <div style="background: rgba(255,255,255,0.8); border-radius: 16px; padding: 1.5rem; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 2rem;">
            <h2 style="color: #2d3748;">Personality and Preferences</h2>
            <p style="color: #4a5568;">
            These questions help us understand your work style, strengths, and preferences.
            </p>
        </div>
        """, unsafe_allow_html=True)

        for i, q in enumerate(st.session_state.questions):
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.9); border-radius: 16px; padding: 1.25rem; 
                            margin-bottom: 1rem; border-left: 4px solid #667eea;">
                    <h3 style="color: #2d3748; margin-bottom: 0.5rem;">{q['question']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                selected_option = st.radio(
                    "",
                    [opt["text"] for opt in q["options"]],
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                selected_value = q["options"][[opt["text"] for opt in q["options"]].index(selected_option)]["value"]
                st.session_state.user_responses[q["feature"]] = selected_value

        # Prediction button with animation
        if st.button("🔮 Find My Career Match", use_container_width=True):
            required_fields = list(direct_input_features.keys()) + ['Interest', 'Work_Style', 'Strengths']
            filled_fields = [field for field in required_fields if field in st.session_state.user_responses]
            
            if len(filled_fields) < 3:
                st.warning("Please answer at least 3 questions (including GPA and Experience) for better results.")
            else:
                with st.spinner("🔍 Analyzing your unique profile..."):
                    try:
                        # Prepare input data
                        input_data = processed_data.drop('Predicted_Career_Field', axis=1).iloc[0:1].copy()
                        
                        le_dict = {}
                        for col in data.select_dtypes(include=['object']).columns:
                            if col in data.columns and col != 'Predicted_Career_Field':
                                le = LabelEncoder()
                                le.fit(data[col].astype(str))
                                le_dict[col] = le

                        # Map user responses
                        for col in input_data.columns:
                            if col in st.session_state.user_responses:
                                if col in ['Communication_Skills', 'Leadership_Skills', 'Teamwork_Skills']:
                                    level_map = {"Low": 0, "Medium": 1, "High": 2}
                                    input_data[col] = level_map.get(st.session_state.user_responses[col], 1)
                                elif col in le_dict:
                                    try:
                                        input_data[col] = le_dict[col].transform([st.session_state.user_responses[col]])[0]
                                    except ValueError:
                                        input_data[col] = processed_data[col].mode()[0]
                                else:
                                    input_data[col] = st.session_state.user_responses[col]
                            else:
                                input_data[col] = processed_data[col].median()
                        
                        # Make prediction
                        prediction = model.predict(input_data)
                        predicted_career = target_le.inverse_transform(prediction)[0]

                        # Display results with animated success card
                        st.success(f"""
                        <div style="border-radius: 16px; padding: 1.5rem; background: rgba(102, 187, 106, 0.9);
                                    color: white; backdrop-filter: blur(5px);">
                            <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
                            🎯 Your Best Career Match
                            </h2>
                            <div style="background: rgba(255,255,255,0.2); border-radius: 12px; padding: 1.5rem; 
                                        text-align: center; margin-bottom: 1rem;">
                                <h1 style="color: white; margin: 0;">{predicted_career}</h1>
                            </div>
                            <p style="text-align: center; margin-bottom: 0;">
                            This career aligns perfectly with your personality and skills!
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Explanation expander
                        with st.expander("💡 Why this career matches you", expanded=True):
                            feat_importances = pd.Series(model.feature_importances_, index=input_data.columns)
                            top_features = feat_importances.sort_values(ascending=False).head(3)
                            
                            cols = st.columns(3)
                            for i, (feat, importance) in enumerate(top_features.items()):
                                with cols[i]:
                                    st.markdown(f"""
                                    <div style="background: rgba(255,255,255,0.9); border-radius: 12px; 
                                                padding: 1rem; text-align: center; height: 100%;">
                                        <h4 style="color: #4a5568; margin-bottom: 0.5rem;">
                                        {feat.replace('_', ' ').title()}
                                        </h4>
                                        <div style="background: linear-gradient(90deg, #667eea, #764ba2); 
                                                    height: 6px; border-radius: 3px; margin: 0.5rem 0;">
                                            <div style="width: {importance*100}%; height: 100%; 
                                                        background: linear-gradient(90deg, #4fd1c5, #48bb78); 
                                                        border-radius: 3px;"></div>
                                        </div>
                                        <p style="color: #4a5568; font-size: 0.9rem; margin-bottom: 0;">
                                        Weight: {importance:.2f}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.write("\nThis career path typically requires these characteristics, which match well with your profile.")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    with tab2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.8); border-radius: 16px; padding: 1.5rem; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 2rem;">
            <h2 style="color: #2d3748;">Career Insights Explorer</h2>
            <p style="color: #4a5568;">
            Explore different career paths and their characteristics in our database.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Career distribution visualization
        st.subheader("📈 Career Path Distribution")
        if 'Predicted_Career_Field' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            career_counts = data['Predicted_Career_Field'].value_counts().head(15)
            ax.barh(career_counts.index, career_counts.values, color=plt.cm.viridis(np.linspace(0, 1, 15)))
            ax.set_title("Top 15 Career Paths in Our Database", fontsize=14, pad=20)
            ax.set_xlabel("Number of Profiles", fontsize=12)
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            st.pyplot(fig)
        else:
            st.warning("No career path data available.")

        # Career details explorer
        st.subheader("🔍 Career Profile Explorer")
        if 'Predicted_Career_Field' in data.columns:
            selected_career = st.selectbox(
                "Select a career to explore:",
                sorted(data['Predicted_Career_Field'].unique()),
                key="career_select"
            )
            
            career_data = data[data['Predicted_Career_Field'] == selected_career]
            
            if not career_data.empty:
                # Stats cards
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.9); border-radius: 16px; padding: 1.5rem; 
                            box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
                    <h3 style="color: #2d3748; margin-top: 0;">{selected_career} Profile</h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                        <div style="background: rgba(102, 187, 106, 0.1); border-radius: 12px; padding: 1rem; 
                                    border: 1px solid rgba(102, 187, 106, 0.3);">
                            <p style="color: #4a5568; font-size: 0.9rem; margin: 0 0 0.5rem 0;">Average GPA</p>
                            <h3 style="color: #2d3748; margin: 0;">{career_data['GPA'].mean():.1f if 'GPA' in career_data.columns and not career_data['GPA'].isnull().all() else 'N/A'}</h3>
                        </div>
                        <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 1rem; 
                                    border: 1px solid rgba(102, 126, 234, 0.3);">
                            <p style="color: #4a5568; font-size: 0.9rem; margin: 0 0 0.5rem 0;">Avg. Experience</p>
                            <h3 style="color: #2d3748; margin: 0;">{career_data['Years_of_Experience'].mean():.1f if 'Years_of_Experience' in career_data.columns else 'N/A'} years</h3>
                        </div>
                        <div style="background: rgba(237, 137, 54, 0.1); border-radius: 12px; padding: 1rem; 
                                    border: 1px solid rgba(237, 137, 54, 0.3);">
                            <p style="color: #4a5568; font-size: 0.9rem; margin: 0 0 0.5rem 0;">Common Interest</p>
                            <h3 style="color: #2d3748; margin: 0;">{career_data['Interest'].mode()[0] if 'Interest' in career_data.columns else 'N/A'}</h3>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
