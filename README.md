# 🎓 Student Performance Predictor

This project uses machine learning to predict whether a student will **pass or fail** based on their:
- Hours Studied
- Attendance
- Hours of Sleep

The goal is to provide a simple model that demonstrates binary classification using real-world educational data.

---

## 📊 Dataset

The dataset contains records of students with the following features:

| Feature         | Description                                |
|-----------------|--------------------------------------------|
| Hours_Studied   | Number of hours studied per week (integer) |
| Attendance      | Class attendance percentage (0–100)        |
| Hours_Sleep     | Average sleep hours per night (integer)    |
| Pass            | 1 = Passed, 0 = Failed                     |

📁 Located at: `student_data_large_int.csv`

---

## 🛠️ How to Run

### 1. Install dependencies

Make sure you have Python 3 installed. Then run:

```bash
pip install -r requirements.txt

