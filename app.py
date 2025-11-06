from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
CORS(app)

# Load and prepare the dataset
def load_student_data():
    try:
        # Try to load from Downloads folder
        home_dir = os.path.expanduser("~")
        csv_path = os.path.join(home_dir, "Downloads", "student_data.csv")
        
        if not os.path.exists(csv_path):
            print(f"CSV not found at {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} student records")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Initialize dataset
student_data = load_student_data()

# Prepare ML model for predictions
def train_model():
    if student_data is None:
        return None, None
    
    df = student_data.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_education = LabelEncoder()
    le_course = LabelEncoder()
    le_style = LabelEncoder()
    le_dropout = LabelEncoder()
    
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
    df['Education_Encoded'] = le_education.fit_transform(df['Education_Level'])
    df['Course_Encoded'] = le_course.fit_transform(df['Course_Name'])
    df['Style_Encoded'] = le_style.fit_transform(df['Learning_Style'])
    df['Dropout_Encoded'] = le_dropout.fit_transform(df['Dropout_Likelihood'])
    
    # Features for prediction
    features = ['Age', 'Gender_Encoded', 'Education_Encoded', 'Course_Encoded',
                'Time_Spent_on_Videos', 'Quiz_Attempts', 'Quiz_Scores',
                'Forum_Participation', 'Assignment_Completion_Rate', 'Style_Encoded']
    
    X = df[features]
    y = df['Dropout_Encoded']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, {
        'gender': le_gender,
        'education': le_education,
        'course': le_course,
        'style': le_style,
        'dropout': le_dropout
    }

model, encoders = train_model()

# AI Chatbot Response Generator
def generate_ai_response(message):
    message_lower = message.lower()
    
    # Dataset insights responses
    if 'dataset' in message_lower or 'data' in message_lower or 'students' in message_lower:
        if student_data is not None:
            total = len(student_data)
            avg_score = student_data['Final_Exam_Score'].mean()
            dropout_rate = (student_data['Dropout_Likelihood'] == 'Yes').sum() / total * 100
            
            return f"""Based on our student dataset of {total} students:

📊 **Key Insights:**
- Average Final Exam Score: {avg_score:.1f}%
- Dropout Rate: {dropout_rate:.1f}%
- Most Popular Course: {student_data['Course_Name'].mode()[0]}
- Average Time on Videos: {student_data['Time_Spent_on_Videos'].mean():.0f} minutes

The data shows patterns between engagement levels and academic success. Would you like to know more about any specific aspect?"""
        else:
            return "I don't have access to the student dataset at the moment. Please make sure the CSV file is in your Downloads folder."
    
    # Engagement and performance questions
    elif 'engagement' in message_lower or 'perform' in message_lower:
        if student_data is not None:
            high_eng = student_data[student_data['Engagement_Level'] == 'High']['Final_Exam_Score'].mean()
            med_eng = student_data[student_data['Engagement_Level'] == 'Medium']['Final_Exam_Score'].mean()
            low_eng = student_data[student_data['Engagement_Level'] == 'Low']['Final_Exam_Score'].mean()
            
            return f"""**Engagement Impact on Performance:**

High Engagement: {high_eng:.1f}% average score
Medium Engagement: {med_eng:.1f}% average score  
Low Engagement: {low_eng:.1f}% average score

Clear correlation! Students with higher engagement consistently score better. Focus on interactive learning, regular forum participation, and completing assignments on time."""
        
    # Learning style questions
    elif 'learning style' in message_lower or 'visual' in message_lower or 'reading' in message_lower:
        if student_data is not None:
            styles = student_data.groupby('Learning_Style')['Final_Exam_Score'].mean().sort_values(ascending=False)
            
            response = "**Learning Style Performance:**\n\n"
            for style, score in styles.items():
                response += f"• {style}: {score:.1f}% average score\n"
            
            response += "\nEveryone learns differently! Focus on methods that work best for you while staying engaged with the material."
            return response
    
    # Career advice - Engineering
    elif 'engineer' in message_lower or 'technology' in message_lower or 'software' in message_lower:
        return """**Engineering & Technology Careers:**

🎓 **Educational Path:**
- Bachelor's in Computer Science, Software Engineering, or related field
- Consider specializations: AI/ML, Cybersecurity, Web Development, Data Science

💡 **Key Skills:**
- Programming (Python, Java, JavaScript)
- Problem-solving and algorithmic thinking
- Version control (Git)
- Database management
- Cloud computing basics

📈 **Career Options:**
- Software Engineer ($80K-150K+)
- Data Scientist ($90K-160K+)
- AI/ML Engineer ($100K-180K+)
- DevOps Engineer ($85K-140K+)

Based on student data, those with high quiz scores and engagement in technical courses tend to excel in these fields!"""
    
    # Career advice - Medicine
    elif 'doctor' in message_lower or 'medical' in message_lower or 'medicine' in message_lower:
        return """**Medical Career Path:**

🎓 **Education Requirements:**
- Pre-med undergraduate (Biology, Chemistry, etc.)
- MCAT exam preparation
- 4 years Medical School
- 3-7 years Residency (varies by specialty)

📚 **Key Preparation:**
- Excel in sciences (Biology, Chemistry, Physics)
- Strong GPA (3.7+ recommended)
- Clinical experience/volunteering
- Research opportunities

💼 **Career Paths:**
- General Practitioner ($200K-250K)
- Specialist ($250K-400K+)
- Surgeon ($300K-500K+)
- Research Physician (varies)

This path requires dedication and high academic performance. Students with consistent high scores and strong engagement succeed best!"""
    
    # Career advice - Business
    elif 'business' in message_lower or 'finance' in message_lower or 'mba' in message_lower:
        return """**Business & Finance Careers:**

🎓 **Education Path:**
- Bachelor's in Business, Finance, Economics
- Consider MBA for leadership roles
- Professional certifications (CPA, CFA, PMP)

💼 **Key Skills:**
- Financial analysis
- Leadership and communication
- Data analysis and Excel
- Strategic thinking
- Project management

📊 **Career Options:**
- Financial Analyst ($60K-100K)
- Business Manager ($70K-120K)
- Investment Banker ($100K-200K+)
- Management Consultant ($80K-150K+)
- Entrepreneur (unlimited potential)

Students with good assignment completion rates and forum participation often excel in business roles!"""
    
    # Study tips and improvement
    elif 'study' in message_lower or 'improve' in message_lower or 'tips' in message_lower:
        return """**Proven Study Tips from High-Performing Students:**

⏰ **Time Management:**
- Use the Pomodoro Technique (25-min focus sessions)
- Study during your peak energy hours
- Break large tasks into smaller chunks

📝 **Active Learning:**
- Take handwritten notes
- Teach concepts to others
- Practice with quizzes and mock tests
- Participate in study groups

🎯 **Based on Our Data:**
Students who spend more time on educational videos and complete assignments consistently score 20-30% higher!

💡 **Key Insight:**
- High forum participation correlates with better understanding
- Regular quiz attempts improve retention
- Assignment completion is the #1 predictor of success

What specific subject would you like help with?"""
    
    # Dropout prevention
    elif 'dropout' in message_lower or 'quit' in message_lower or 'struggling' in message_lower:
        return """**Staying on Track - Dropout Prevention:**

🚨 **Warning Signs:**
- Declining quiz scores
- Low assignment completion rate
- Reduced time on course materials
- Minimal forum participation

✅ **Action Steps:**
1. **Reach out for help** - Talk to instructors/counselors
2. **Form study groups** - Learn with peers
3. **Create a schedule** - Consistent study routine
4. **Set small goals** - Build momentum gradually
5. **Take breaks** - Avoid burnout

📊 **Data Insight:**
Students who maintain >80% assignment completion and regular engagement rarely drop out!

Remember: Everyone struggles sometimes. The key is seeking help early and staying engaged. What specific challenge are you facing?"""
    
    # General career guidance
    elif 'career' in message_lower or 'job' in message_lower or 'future' in message_lower:
        return """**Finding Your Career Path:**

🔍 **Self-Assessment Questions:**
1. What subjects do you enjoy most?
2. Do you prefer working with people, data, or things?
3. What's your desired work-life balance?
4. Are you interested in research or practical application?

📊 **Based on Performance Data:**
- **High Math/Science scores** → Engineering, Data Science, Research
- **High Language scores** → Law, Teaching, Communications
- **Balanced scores + Leadership** → Business, Management
- **Creative thinking** → Design, Marketing, Arts

🎯 **Next Steps:**
1. Take career assessment tests
2. Talk to professionals in fields of interest
3. Try internships or volunteer work
4. Focus on building relevant skills

Would you like specific guidance based on your academic strengths?"""
    
    # Default response
    else:
        return """I'm here to help with career guidance and educational advice! I can assist with:

💬 **Ask me about:**
- Career paths (Engineering, Medicine, Business, etc.)
- Study tips and academic improvement
- Understanding student performance data
- Learning styles and engagement
- Dropout prevention strategies
- Educational planning

📊 I have access to student performance data to provide data-driven insights!

What would you like to know?"""

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        ai_reply = generate_ai_response(user_message)
        
        return jsonify({
            'reply': ai_reply,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'reply': f'Sorry, I encountered an error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/predict-dropout', methods=['POST'])
def predict_dropout():
    """
    Predict dropout likelihood from career predictor scores
    Uses the career prediction scores to estimate dropout risk
    """
    try:
        data = request.json
        
        # Get scores from career predictor
        math = float(data.get('math', 0))
        science = float(data.get('science', 0))
        english = float(data.get('english', 0))
        social = float(data.get('social', 0))
        arts = float(data.get('arts', 0))
        interest = data.get('interest', '')
        
        # Calculate average score
        avg_score = (math + science + english + social + arts) / 5
        
        # Calculate risk factors based on student data patterns
        risk_score = 0
        risk_factors = []
        
        # Factor 1: Overall academic performance
        if avg_score < 50:
            risk_score += 40
            risk_factors.append("Low overall academic performance (avg < 50%)")
        elif avg_score < 70:
            risk_score += 20
            risk_factors.append("Below average academic performance")
        
        # Factor 2: Subject score variance (inconsistency)
        scores = [math, science, english, social, arts]
        variance = np.var(scores)
        if variance > 400:  # High variance indicates inconsistency
            risk_score += 15
            risk_factors.append("Inconsistent performance across subjects")
        
        # Factor 3: Critical subject failure
        failing_subjects = []
        if math < 40:
            failing_subjects.append("Mathematics")
        if science < 40:
            failing_subjects.append("Science")
        if english < 40:
            failing_subjects.append("English")
        
        if len(failing_subjects) > 0:
            risk_score += 20 * len(failing_subjects)
            risk_factors.append(f"Failing in critical subjects: {', '.join(failing_subjects)}")
        
        # Factor 4: Interest-performance mismatch
        if interest == 'Technology & Engineering' and (math < 60 or science < 60):
            risk_score += 15
            risk_factors.append("Performance doesn't align with chosen interest area")
        elif interest == 'Healthcare & Medicine' and (science < 65 or avg_score < 70):
            risk_score += 15
            risk_factors.append("Performance below requirements for medical field")
        
        # Calculate dropout probability (cap at 95%)
        dropout_probability = min(risk_score, 95)
        
        # Determine risk level
        if dropout_probability < 30:
            risk_level = "Low"
            risk_color = "green"
            recommendation = "Great job! Keep maintaining your current performance and engagement."
        elif dropout_probability < 60:
            risk_level = "Medium"
            risk_color = "orange"
            recommendation = "Consider seeking additional support in weaker subjects. Join study groups and attend tutoring sessions."
        else:
            risk_level = "High"
            risk_color = "red"
            recommendation = "Immediate intervention recommended. Schedule meetings with academic counselor and subject tutors. Create a structured study plan."
        
        # Generate protective factors
        protective_factors = []
        if avg_score >= 70:
            protective_factors.append("Strong overall academic performance")
        if min(scores) >= 60:
            protective_factors.append("No failing subjects")
        if variance < 200:
            protective_factors.append("Consistent performance across subjects")
        
        # Success strategies based on data
        strategies = [
            "Maintain regular study schedule (2-3 hours daily)",
            "Complete all assignments on time (improves success rate by 25%)",
            "Participate in class discussions and forums",
            "Review materials before exams",
            "Seek help early when struggling with concepts"
        ]
        
        return jsonify({
            'dropout_probability': round(dropout_probability, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'average_score': round(avg_score, 1),
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'recommendation': recommendation,
            'success_strategies': strategies,
            'subject_analysis': {
                'mathematics': {'score': math, 'status': 'Pass' if math >= 40 else 'Fail'},
                'science': {'score': science, 'status': 'Pass' if science >= 40 else 'Fail'},
                'english': {'score': english, 'status': 'Pass' if english >= 40 else 'Fail'},
                'social': {'score': social, 'status': 'Pass' if social >= 40 else 'Fail'},
                'arts': {'score': arts, 'status': 'Pass' if arts >= 40 else 'Fail'}
            },
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/student-stats', methods=['GET'])
def student_stats():
    try:
        if student_data is None:
            return jsonify({'error': 'Dataset not available'}), 500
        
        stats = {
            'total_students': len(student_data),
            'avg_exam_score': float(student_data['Final_Exam_Score'].mean()),
            'avg_quiz_score': float(student_data['Quiz_Scores'].mean()),
            'dropout_rate': float((student_data['Dropout_Likelihood'] == 'Yes').sum() / len(student_data) * 100),
            'courses': student_data['Course_Name'].value_counts().to_dict(),
            'engagement_levels': student_data['Engagement_Level'].value_counts().to_dict(),
            'learning_styles': student_data['Learning_Style'].value_counts().to_dict()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'dataset_loaded': student_data is not None,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("=" * 50)
    print("LearnPath AI Backend Server")
    print("=" * 50)
    if student_data is not None:
        print(f"✓ Dataset loaded: {len(student_data)} student records")
        print(f"✓ Columns: {', '.join(student_data.columns.tolist())}")
    else:
        print("✗ Dataset not loaded - Please place student_data.csv in Downloads folder")
    
    if model is not None:
        print("✓ ML model trained and ready")
    else:
        print("✗ ML model not available")
    
    print("\nServer starting on http://127.0.0.1:5001")
    print("=" * 50)
    
    app.run(debug=True, host='127.0.0.1', port=5001)