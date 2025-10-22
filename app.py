from flask import Flask, request, render_template_string, send_file
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import os, io
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --------------------- Setup ---------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
nlp = spacy.load("en_core_web_sm")

canonical_skills = ["python","java","c++","sql","machine learning","deep learning",
                    "data analysis","statistics","pandas","numpy","react","javascript",
                    "html","css","git","data structures","algorithms","docker","kubernetes",
                    "aws","linux","nodejs","django","flask","tensorflow","pytorch","excel"]

ROLE_SKILLS = {
    "data scientist":["python","machine learning","data analysis","statistics","pandas","numpy","tensorflow","pytorch"],
    "ml engineer":["python","machine learning","deep learning","tensorflow","pytorch","numpy","pandas"],
    "data analyst":["python","sql","excel","statistics","data analysis","pandas","numpy"],
    "frontend developer":["javascript","react","html","css"],
    "backend developer":["python","django","flask","sql","git","nodejs"],
    "full stack developer":["javascript","react","html","css","python","django","flask","sql","git"],
    "sde":["java","c++","data structures","algorithms","git"],
    "devops engineer":["docker","kubernetes","aws","linux","git"]
}

SKILL_RESOURCES = {
    "python":"https://www.learnpython.org/","java":"https://www.sololearn.com/Course/Java/",
    "c++":"https://www.sololearn.com/Course/CPlusPlus/","sql":"https://www.khanacademy.org/computing/computer-programming/sql",
    "machine learning":"https://www.coursera.org/learn/machine-learning",
    "deep learning":"https://www.coursera.org/specializations/deep-learning",
    "data analysis":"https://www.kaggle.com/learn/data-analysis",
    "statistics":"https://www.khanacademy.org/math/statistics-probability",
    "pandas":"https://www.datacamp.com/courses/pandas-foundations",
    "numpy":"https://numpy.org/learn/","react":"https://reactjs.org/tutorial/tutorial.html",
    "javascript":"https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/",
    "html":"https://www.freecodecamp.org/learn/responsive-web-design/",
    "css":"https://www.freecodecamp.org/learn/responsive-web-design/","git":"https://www.learnenough.com/git-tutorial",
    "data structures":"https://www.geeksforgeeks.org/data-structures/","algorithms":"https://www.geeksforgeeks.org/fundamentals-of-algorithms/",
    "docker":"https://www.docker.com/get-started","kubernetes":"https://www.kubernetes.io/docs/tutorials/",
    "aws":"https://aws.amazon.com/training/","linux":"https://linuxjourney.com/",
    "nodejs":"https://nodejs.org/en/docs/guides/","django":"https://docs.djangoproject.com/en/4.2/intro/tutorial01/",
    "flask":"https://flask.palletsprojects.com/en/2.3.x/tutorial/","tensorflow":"https://www.tensorflow.org/tutorials",
    "pytorch":"https://pytorch.org/tutorials/","excel":"https://support.microsoft.com/en-us/excel"
}

SKILL_PROJECTS = {
    "python": ["Build a web scraper", "Automate file handling scripts"],
    "java": ["Create a banking system app", "Implement a library management system"],
    "c++": ["Build a console-based game", "Implement data structures from scratch"],
    "sql": ["Create a small inventory database", "Analyze sales data with queries"],
    "machine learning": ["Predict housing prices using ML", "Classify images with sklearn"],
    "deep learning": ["Train a CNN on MNIST dataset", "Image classification using TensorFlow"],
    "react": ["Build a personal portfolio site", "Create a to-do web app"],
    "flask": ["Develop a simple blog web app", "Build a REST API for a small project"],
    "docker": ["Containerize a web app", "Dockerize a Python project"],
    "aws": ["Deploy a web app on AWS EC2", "Set up S3 bucket for file hosting"]
}

# --------------------- Utilities ---------------------
def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        return extract_pdf_text(file_path)
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def extract_skills_nlp(text):
    doc = nlp(text.lower())
    return list(set([token.text for token in doc if token.text in canonical_skills]))

def highlight_skills(text, skills):
    for skill in skills:
        text = text.replace(skill, f"<mark>{skill}</mark>")
    return text

def calculate_fit_percentage(skills_found, target_skills):
    if not target_skills:
        return 0
    return round(len(set(skills_found) & set(target_skills)) / len(target_skills) * 100, 1)

def generate_insight(role, fit, missing_skills):
    if fit > 80:
        return f"You are highly qualified for the {role} role! Keep refining your skills."
    elif 50 <= fit <= 80:
        return f"You are partially ready for the {role} role. Focus on learning {', '.join(missing_skills[:3])}."
    else:
        return f"You're just starting for the {role} role. Begin with {', '.join(missing_skills[:3])}."

# --------------------- ML Role Prediction ---------------------
vectorizer = TfidfVectorizer(max_features=5000)
clf = LogisticRegression()
X_train = [
    "python machine learning pandas numpy statistics tensorflow pytorch",
    "javascript react html css frontend",
    "java c++ data structures algorithms git",
    "python django flask sql git",
    "docker kubernetes aws linux git"
]
y_train = ["data scientist","frontend developer","sde","backend developer","devops engineer"]
X_train_tfidf = vectorizer.fit_transform(X_train)
clf.fit(X_train_tfidf, y_train)

# --------------------- Routes ---------------------
@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>AI Resume Analyzer</title>
<link href="https://bootswatch.com/5/minty/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-4 bg-light">
<div class="container text-center">
<h2 class="mb-4">🤖 AI Resume & Skill Gap Analyzer</h2>
<div class="card p-4 shadow-lg">
<form method="POST" action="/analyze" enctype="multipart/form-data">
<div class="mb-3 text-start">
<label class="form-label">Upload Resume (PDF/DOCX)</label>
<input type="file" name="resume_file" class="form-control" required>
</div>
<div class="mb-3 text-start">
<label class="form-label">Target Role</label>
<select name="role" class="form-select" required>
{% for role in roles %}
<option value="{{ role }}">{{ role.title() }}</option>
{% endfor %}
</select>
</div>
<button class="btn btn-success w-100" type="submit">Analyze Resume</button>
</form>
</div>
</div>
</body>
</html>
""", roles=ROLE_SKILLS.keys())

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get("resume_file")
    role_selected = request.form.get("role")
    if not file:
        return "No file uploaded!"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    text = extract_text(file_path)

    skills_found = extract_skills_nlp(text)
    target_skills = ROLE_SKILLS.get(role_selected.lower(), [])
    missing_skills = [s for s in target_skills if s not in skills_found]
    fit_percentage = calculate_fit_percentage(skills_found, target_skills)
    insight = generate_insight(role_selected, fit_percentage, missing_skills)
    highlighted_resume = highlight_skills(text.lower(), skills_found)

    # ML predicted role
    X_test = vectorizer.transform([text])
    predicted_role = clf.predict(X_test)[0]

    chart_data = {
        "labels": target_skills,
        "datasets":[{"label":"Skill Coverage",
                     "data":[1 if s in skills_found else 0 for s in target_skills],
                     "backgroundColor":["#28a745" if s in skills_found else "#dc3545" for s in target_skills]}]
    }

    missing_projects = {skill: SKILL_PROJECTS.get(skill, ["No project suggestion"]) for skill in missing_skills}

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>Resume Analysis Result</title>
<link href="https://bootswatch.com/5/minty/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>mark{background:yellow;}</style>
</head>
<body class="p-4 bg-light">
<div class="container">
<div class="card shadow-lg p-4">
<h3 class="mb-3 text-center">{{ role_selected.title() }} Role Analysis</h3>

<p><b>Predicted Role:</b> {{ predicted_role.title() }}</p>
<p><b>Fit Percentage:</b> {{ fit_percentage }}%</p>

<div class="progress mb-3" style="height:25px;">
  <div class="progress-bar bg-success" style="width:{{ fit_percentage }}%;">{{ fit_percentage }}%</div>
</div>

<div class="alert alert-info"><b>AI Insight:</b> {{ insight }}</div>

<div class="row">
<div class="col-md-6">
<h5>✅ Current Skills</h5>
<ul class="list-group">
{% for skill in skills_found %}
<li class="list-group-item list-group-item-success">{{ skill }}</li>
{% endfor %}
</ul>

<h5 class="mt-4">❌ Missing Skills & Resources</h5>
<ul class="list-group">
{% for skill in missing_skills %}
<li class="list-group-item list-group-item-danger">
<b>{{ skill }}</b> - <a href="{{ resources[skill] }}" target="_blank">Learn</a>
<ul>
{% for proj in missing_projects[skill] %}
<li>{{ proj }}</li>
{% endfor %}
</ul>
</li>
{% endfor %}
</ul>
</div>

<div class="col-md-6">
<h5>📊 Skill Coverage</h5>
<canvas id="chart"></canvas>
<h5 class="mt-4">📝 Highlighted Resume</h5>
<div class="border p-2 bg-white" style="max-height:300px;overflow:auto;">{{ highlighted_resume | safe }}</div>
</div>
</div>

<a href="/" class="btn btn-secondary mt-4">Go Back</a>
<a href="/download_report?role={{ role_selected }}&fit={{ fit_percentage }}" class="btn btn-primary mt-4">Download PDF Report</a>
</div>
</div>

<script>
const ctx=document.getElementById('chart').getContext('2d');
new Chart(ctx,{type:'bar',data:{{ chart_data | tojson }},options:{scales:{y:{beginAtZero:true,ticks:{stepSize:1}}}}});
</script>
</body>
</html>
""", role_selected=role_selected, predicted_role=predicted_role,
       fit_percentage=fit_percentage, insight=insight,
       skills_found=skills_found, missing_skills=missing_skills,
       resources=SKILL_RESOURCES, missing_projects=missing_projects,
       chart_data=chart_data, highlighted_resume=highlighted_resume)

@app.route('/download_report')
def download_report():
    role = request.args.get("role", "Unknown Role")
    fit = request.args.get("fit", "0")

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, 800, "AI Resume Analyzer Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 770, f"Target Role: {role.title()}")
    p.drawString(100, 750, f"Fit Percentage: {fit}%")
    p.drawString(100, 730, f"Generated by AI Resume Analyzer")
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="resume_analysis_report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True)


