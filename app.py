import pandas as pd
import numpy as np
import random
import re
from datetime import datetime, timedelta

# Flask for web server
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Machine Learning & NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# --- Download NLTK data once ---
nltk.download('punkt')

# Download the 'stopwords' corpus
nltk.download('stopwords')

# --- GLOBAL SETUP: We will use your original class directly ---
# This makes the setup cleaner.

class TaskManagementSystem:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.category_model = SVC(kernel='linear', probability=True, random_state=42)
        self.priority_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        # Encoders and Scalers need to be stored
        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        self.priority_feature_scaler = StandardScaler()
        self.priority_feature_columns = []

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def generate_synthetic_dataset(self, n_samples=1000):
        # This is your original dataset generation function, kept intact.
        categories = ['Development', 'Testing', 'Documentation', 'Design', 'Marketing', 'Research', 'Bug Fix', 'Feature']
        priorities = ['Low', 'Medium', 'High', 'Critical']
        statuses = ['To Do', 'In Progress', 'Review', 'Done']
        task_templates = {'Development':['Implement user authentication system','Create REST API endpoints','Develop frontend components'],'Testing':['Write unit tests for user service','Perform integration testing on API'],'Documentation':['Update API documentation','Create user manual for new features'],'Design':['Design user interface mockups','Create brand identity guidelines'],'Marketing':['Launch social media campaign','Create email marketing templates'],'Research':['Research competitor analysis','Investigate new technology trends'],'Bug Fix':['Fix login authentication issue','Resolve database connection errors'],'Feature':['Add dark mode toggle','Implement search functionality']}
        data = []
        for i in range(n_samples):
            category = random.choice(categories)
            description = random.choice(task_templates[category])
            if random.random() < 0.3: description += f" ({random.choice(['urgent','critical','asap','important'])})"
            priority_weights = {'Bug Fix': [0.1, 0.2, 0.4, 0.3],'Development': [0.2, 0.4, 0.3, 0.1],'Testing': [0.3, 0.4, 0.2, 0.1],'Documentation': [0.4, 0.4, 0.15, 0.05],'Design': [0.3, 0.4, 0.25, 0.05],'Marketing': [0.25, 0.45, 0.25, 0.05],'Research': [0.35, 0.45, 0.15, 0.05],'Feature': [0.2, 0.4, 0.35, 0.05]}
            priority_idx = np.random.choice(4, p=priority_weights[category])
            if any(word in description.lower() for word in ['urgent', 'critical']): priority_idx = min(3, priority_idx + 1)
            priority = priorities[priority_idx]
            created_date = datetime.now() - timedelta(days=random.randint(1, 90))
            data.append({'task_id':f"TASK-{i+1:04d}",'description':description,'category':category,'priority':priority,'assignee':f"user_{random.randint(1, 20)}",'reporter':f"user_{random.randint(1, 15)}",'status':random.choice(statuses),'created_date':created_date.strftime('%Y-%m-%d'),'due_date':(created_date + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),'effort_points':random.choice([1,2,3,5,8,13]),'workload_percentage':random.randint(20, 80)})
        return pd.DataFrame(data)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = [self.stemmer.stem(token) for token in text.split() if token not in self.stop_words]
        return ' '.join(tokens)

    def train_models(self, df):
        print("🤖 Training models...")
        # --- Train Category Classifier ---
        df['processed_description'] = df['description'].apply(self.preprocess_text)
        X_cat = self.tfidf_vectorizer.fit_transform(df['processed_description'])
        y_cat = self.category_encoder.fit_transform(df['category'])
        self.category_model.fit(X_cat, y_cat)
        print("✅ Category model trained.")

        # --- Train Priority Predictor ---
        df['description_length'] = df['description'].str.len()
        df['word_count'] = df['description'].str.split().str.len()
        df['category_encoded'] = y_cat # Reuse encoded category
        
        self.priority_feature_columns = ['description_length', 'word_count', 'effort_points', 'category_encoded']
        X_prio = df[self.priority_feature_columns]
        y_prio = self.priority_encoder.fit_transform(df['priority'])
        
        X_prio_scaled = self.priority_feature_scaler.fit_transform(X_prio)
        self.priority_model.fit(X_prio_scaled, y_prio)
        print("✅ Priority model trained.")
        
        return df

    # In app.py, inside the TaskManagementSystem class...

    def predict_task(self, description):
        # --- This first part is the same as before ---
        # Preprocess text
        processed_desc = self.preprocess_text(description)
        desc_vector = self.tfidf_vectorizer.transform([processed_desc])
        
        # Predict category
        category_encoded = self.category_model.predict(desc_vector)[0]
        predicted_category = self.category_encoder.inverse_transform([category_encoded])[0]
        
        # Prepare features for priority prediction
        priority_features = pd.DataFrame({
            'description_length': [len(description)],
            'word_count': [len(description.split())],
            'effort_points': [5], # Use a default average effort for new tasks
            'category_encoded': [category_encoded]
        })
        priority_features = priority_features[self.priority_feature_columns]
        priority_features_scaled = self.priority_feature_scaler.transform(priority_features)
        
        # Predict priority
        priority_encoded = self.priority_model.predict(priority_features_scaled)[0]
        predicted_priority = self.priority_encoder.inverse_transform([priority_encoded])[0]

        # --- NEW LOGIC STARTS HERE ---

        # 1. Assign to a random team member for this demo
        team_members = [f'user_{i}' for i in range(1, 21)]
        predicted_assignee = random.choice(team_members)

        # 2. Estimate effort points based on priority
        effort_map = {'Low': [1, 2, 3], 'Medium': [3, 5, 8], 'High': [8, 13], 'Critical': [13, 21]}
        predicted_effort = random.choice(effort_map[predicted_priority])

        # 3. Calculate a suggested due date based on priority
        days_to_add = 14 # Default for Low/Medium
        if predicted_priority == 'Critical':
            days_to_add = random.randint(1, 3)
        elif predicted_priority == 'High':
            days_to_add = random.randint(3, 7)
        
        due_date = datetime.now() + timedelta(days=days_to_add)
        predicted_due_date = due_date.strftime('%Y-%m-%d')
        
        # 4. Return the complete dictionary with all the new information
        return {
            'category': predicted_category, 
            'priority': predicted_priority,
            'assignee': predicted_assignee,
            'effort': predicted_effort,
            'due_date': predicted_due_date
        }
# --- GLOBAL SETUP ---
# Initialize the system and train the models once when the server starts.
print("🚀 Initializing AI Task Management System...")
tms = TaskManagementSystem()
dataframe = tms.generate_synthetic_dataset()
dataframe = tms.train_models(dataframe)
print("✅ System ready.")

# --- FLASK WEB SERVER ---
app = Flask(__name__)
CORS(app)

# Route to serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to provide data for the dashboard
@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data():
    response = {
        'metrics': {
            'totalTasks': len(dataframe),
            'completedTasks': int(dataframe[dataframe['status'] == 'Done'].shape[0]),
            'highPriorityTasks': int(dataframe[dataframe['priority'].isin(['High', 'Critical'])].shape[0]),
            'aiAccuracy': 0.92 # Placeholder
        },
        'charts': {
            'categories': dataframe['category'].value_counts().to_dict(),
            'priorities': dataframe['priority'].value_counts().to_dict(),
        }
    }
    return jsonify(response)

# Route to handle live predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    description = data.get('description', '')
    if not description:
        return jsonify({'error': 'Description is required'}), 400
    
    prediction = tms.predict_task(description)
    return jsonify(prediction)

# --- START THE SERVER ---
if __name__ == '__main__':
    print("🌍 Starting local Flask server at http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop the server.")
    app.run(port=5000, debug=False)
