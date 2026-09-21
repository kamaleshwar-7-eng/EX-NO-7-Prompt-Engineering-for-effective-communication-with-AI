# Ex.No.7 – Prompt Engineering for Effective Communication with AI

## AIM

To understand and apply the principles of **Prompt Engineering** for effective communication with Artificial Intelligence and analyze how structured instructions can improve the accuracy, relevance, and usefulness of AI-generated outputs.

---

## PROCEDURE

### Step 1: Import Required Libraries

- Import Pandas for creating and manipulating the dataset.
- Import NumPy for numerical operations and random data generation.
- Import Matplotlib for visualization.
- Import Seaborn for creating statistical plots.

### Step 2: Generate a Sample Dataset

- Set a random seed to ensure reproducible results.
- Generate a sample dataset containing 1000 records.
- Create demographic groups based on:
  - Age Group
  - Income Level
- Generate creditworthiness values.
- Generate loan approval outcomes.
- Store all generated information in a Pandas DataFrame.

### Step 3: Define True Positive Rate Calculation

- Create a function to calculate the **True Positive Rate (TPR)**.
- Divide the data according to a selected demographic group.
- Identify True Positives (TP).
- Identify False Negatives (FN).
- Calculate TPR using:

`TPR = TP / (TP + FN)`

### Step 4: Calculate TPR for Age Groups

- Apply the TPR function to the `AgeGroup` column.
- Calculate the True Positive Rate for each age group.
- Store the results in a separate DataFrame.

### Step 5: Calculate TPR for Income Levels

- Apply the TPR function to the `IncomeLevel` column.
- Calculate the True Positive Rate for each income category.
- Store the results in a separate DataFrame.

### Step 6: Visualize TPR Results

- Create bar charts for the TPR values.
- Display TPR values for different age groups.
- Display TPR values for different income levels.
- Add numerical TPR values above the bars.

### Step 7: Identify Potential Bias

- Find the maximum and minimum TPR for age groups.
- Find the maximum and minimum TPR for income levels.
- Calculate the difference between maximum and minimum TPR.
- If the difference is greater than 0.1, display a message indicating a potential disparity in TPRs.

---

## THEORY

### 1. Prompt Engineering

Prompt Engineering is the process of designing and structuring instructions given to an Artificial Intelligence model to obtain accurate, relevant, and useful responses.

A well-designed prompt provides the AI with sufficient context, instructions, constraints, and expected output format.

### 2. Importance of Prompt Engineering

Prompt engineering helps users communicate effectively with AI systems.

It can be used to:

- Improve response accuracy.
- Provide relevant context.
- Control the response format.
- Control tone and level of detail.
- Handle complex tasks.
- Convert vague requirements into structured instructions.

### 3. Clear Instructions

A prompt should clearly explain what the AI needs to do.

For example:

`Explain CNN in simple terms for a first-year engineering student.`

This provides both the task and the expected level of explanation.

### 4. Providing Context

Context gives the AI additional information required to understand the task correctly.

For example:

`Explain Python programming for an ECE engineering student with basic programming knowledge.`

The additional context helps define the intended audience.

### 5. Specifying Output Format

The expected format can be included in the prompt.

Examples include:

- Bullet points
- Tables
- Step-by-step explanations
- JSON
- Code
- Reports

Example:

`Compare CNN and RNN in a table with five differences.`

### 6. Role-Based Prompting

Role-based prompting assigns a particular role or perspective to the AI.

Example:

`Act as a Python programming instructor and explain exception handling with examples.`

This can help structure the response according to the requested context.

### 7. Few-Shot Prompting

Few-shot prompting provides examples of the expected input-output behavior before giving the actual task.

Example:

`Input: 2 → Even`
`Input: 5 → Odd`
`Input: 8 → ?`

The examples help demonstrate the expected pattern.

### 8. Zero-Shot Prompting

Zero-shot prompting asks the AI to perform a task without providing examples.

Example:

`Classify the following sentence as positive or negative: "The product is excellent."`

### 9. Breaking Complex Tasks into Steps

Complex tasks can be divided into smaller instructions.

For example:

1. Read the dataset.
2. Clean the data.
3. Calculate statistics.
4. Generate a visualization.
5. Explain the result.

This makes the requested workflow clearer.

### 10. Controlling Tone and Length

Prompts can specify how the response should be written.

Examples:

`Explain in simple language.`

`Give the answer in 100 words.`

`Write the answer in a formal academic style.`

### 11. Fairness and Bias

AI systems can produce biased outcomes when data or decision processes contain disparities.

In this experiment, **True Positive Rate (TPR)** is calculated across different demographic groups to examine whether the rate of correctly identifying creditworthy applicants differs between groups.

### 12. True Positive Rate

True Positive Rate measures the proportion of actual positive cases that are correctly identified.

In this experiment:

- Positive = Creditworthy
- True Positive = Creditworthy and Approved
- False Negative = Creditworthy and Not Approved

Formula:

`TPR = TP / (TP + FN)`

Comparing TPR across groups can help identify disparities that may require further investigation.

---

## PROGRAM

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Generate a sample dataset
np.random.seed(42)
data_size = 1000
# Demographic groups and loan approval outcomes
age_group = np.random.choice(['Under 30', '30-50', 'Over 50'], size=data_size, p=[0.3, 0.5, 0.2])
income_level = np.random.choice(['Low', 'Medium', 'High'], size=data_size, p=[0.4, 0.4, 0.2])
creditworthy = np.random.choice([1, 0], size=data_size, p=[0.7, 0.3])
 # 1: Creditworthy, 0: Not creditworthy
approved = np.random.choice([1, 0], size=data_size, p=[0.6, 0.4])
  # 1: Approved, 0: Not approved
loan_df = pd.DataFrame({'AgeGroup': age_group, 'IncomeLevel': income_level, 'Creditworthy': creditworthy, 'Approved': approved})
# Function to calculate TPR
def calculate_tpr(df, group_col):
    tpr_data = []
    groups = df[group_col].unique()
    for group in groups:
        group_data = df[df[group_col] == group]
        tp = ((group_data['Creditworthy'] == 1) & (group_data['Approved'] == 1)).sum()
        fn = ((group_data['Creditworthy'] == 1) & (group_data['Approved'] == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_data.append({'Group': group, 'TPR': tpr})
    return pd.DataFrame(tpr_data)

# Calculate TPR for Age Groups
tpr_age = calculate_tpr(loan_df, 'AgeGroup')
# Calculate TPR for Income Levels
tpr_income = calculate_tpr(loan_df, 'IncomeLevel')

# Plotting the TPR for different groups
# Plotting the TPR for different groups
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
sns.barplot(data=tpr_age, x='Group', y='TPR', palette='viridis', hue='Group', legend=False)
plt.title('True Positive Rate by Age Group')
plt.ylabel('True Positive Rate')
plt.xlabel('Age Group')
for i, tpr in enumerate(tpr_age['TPR']):
    plt.text(i, tpr + 0.02, f'{tpr:.2f}', ha='center', va='bottom')
plt.subplot(1, 2, 2)
sns.barplot(data=tpr_income, x='Group', y='TPR', palette='viridis', hue='Group', legend=False)
plt.title('True Positive Rate by Income Level')
plt.ylabel('True Positive Rate')
plt.xlabel('Income Level')
for i, tpr in enumerate(tpr_income['TPR']):
    plt.text(i, tpr + 0.02, f'{tpr:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
# Highlighting potential bias
max_tpr_age = tpr_age['TPR'].max()
min_tpr_age = tpr_age['TPR'].min()
max_tpr_income = tpr_income['TPR'].max()
min_tpr_income = tpr_income['TPR'].min()
if max_tpr_age - min_tpr_age > 0.1:
    print(f"Potential bias detected in Age Group TPRs: Max TPR = {max_tpr_age:.2f}, Min TPR = {min_tpr_age:.2f}")
if max_tpr_income - min_tpr_income > 0.1:
    print(f"Potential bias detected in Income Level TPRs: Max TPR = {max_tpr_income:.2f}, Min TPR = {min_tpr_income:.2f}")
```

## OUTPUT

<img width="1672" height="741" alt="image" src="https://github.com/user-attachments/assets/46f31560-c5c2-4d74-98b1-ff2e072c3f13" />


## CONCLUSION

Thus, the principles of **Prompt Engineering** were studied and applied to understand how clear, structured, and contextual instructions can improve communication with AI systems. The experiment also demonstrated how AI-related decision processes can be examined for disparities by comparing True Positive Rates across demographic groups. Prompt engineering provides a structured approach for obtaining useful and well-formatted AI outputs.
