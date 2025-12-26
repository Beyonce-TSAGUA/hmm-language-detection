# 🌍🔤 Automatic Language Detection with Hidden Markov Models (HMM)

## 📌 Project Overview
This project implements an automatic language recognition system based on **Hidden Markov Models (HMMs)**.  
The goal: **identify the language of a word or text** by exploiting statistical regularities in sequences of letters.

This repository highlights skills in **probabilistic modeling, algorithmics, and Python**, in contexts close to real-world **Natural Language Processing (NLP)** problems.

---

## 🎯 Technical Objectives
- 🧠 Implement a **probabilistic HMM model from scratch**
- 🔡 Analyze **character sequences** for language classification
- ⚖️ Compare different modeling strategies and measure their **performance**
- 📝 Produce a **critical analysis** of the results

---

## 🧩 Key Skills Demonstrated
- 📊 **Statistical modeling (HMM)**
- 🔁 **Probabilistic algorithms**: Forward / Backward
- 🧬 **Sequence analysis**
- 🧮 **Matrix computation & linear algebra**
- 📉 **Model evaluation** (confusion matrices)
- 🐍 **Scientific Python programming**

---

## 🛠️ Tools & Technologies
- 🐍 **Python**
- 🔢 **NumPy** – matrix computations
- 🗂️ **Pandas** – data manipulation
- 📈 **Matplotlib** – visualization
- ⚙️ **SciPy** – numerical tools

---

## 🧪 Methodology

### 1️⃣ Data Preprocessing
- 🧹 Cleaning textual corpora
- 🔤 Normalization (lowercase, remove accents/special characters)
- 🔁 Convert words into **letter sequences (a–z)**

### 2️⃣ HMM Model Construction
Each language is represented by a distinct HMM:

- 🔀 **Transition matrix**: probability of moving between letters
- 🎯 **Emission matrix**: probability of emitting symbols
- 🚀 **Initial probability vector**

### 3️⃣ Probabilistic Inference
- ⚙️ Implement **Forward and Backward algorithms**
- 📊 Calculate the probability that a word/text belongs to a language
- 🏆 Select the **most probable language**

### 4️⃣ Evaluation & Analysis
- 🧪 Classification **word by word** and **text by text**
- 🧩 Build **confusion matrices**
- 🔍 Analyze the impact of:
  - Word length
  - Internal sequence structure
  - Emission matrix

---

## ⭐ Key Results
- 📏 Long words are classified **much more accurately**
- ❓ Short words are **more ambiguous**
- 🎯 Emission matrix strongly affects performance
- ⚠️ Identity emission matrix → **significant drop in accuracy**

---

## 💼 Value for Recruiters
This project demonstrates:

- 🧠 Ability to implement **complex mathematical models**
- 📚 Solid understanding of **probabilistic foundations**
- 🧪 Rigorous approach to **model evaluation**
- 🧐 Skill in **analyzing and explaining system limitations**
- 🚀 Transferable skills for **Machine Learning, NLP, and Data Science**

---

## 🚀 Potential Improvements
- 📚 Enrich the training corpora
- 🌍 Add **new languages**
- ⚙️ Optimize **model parameters**
- 🤖 Introduce **learning algorithms** (Baum-Welch)

---

## ✍️ Author
**TSAGUA YEMEWA Beyoncé**
