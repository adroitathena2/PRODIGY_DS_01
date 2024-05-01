import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("C:\\Users\\archi\\Documents\\INTERNSHIPS\\PRODIGY INFOTECH\\Task 1\\Employee Attrition\\data.csv")
print(df)
print(df.head())
print(df.tail())
print(df.columns)
print(df.shape)
print(df.size)
print(df.index)
df.describe()
df1=df.dropna()
df1.head()
df1.isna().sum()
df1.info()

#https://www.kaggle.com/code/janiobachmann/attrition-in-an-organization-why-workers-quit/notebook

#____________________________________________________________________

# 1. Education Field wise attrition plot
# Group data by EducationField and Attrition to get counts
education_field_attrition = df1.groupby(["EducationField", "Attrition"]).size().reset_index(name='count')
# Create a bar plot showing attrition by education field
plt.figure(figsize=(12, 6))
sns.barplot(x="EducationField", y="count", hue="Attrition", data=education_field_attrition, palette=["#0097F1", "#F9C548"])
# Set labels and title for the plot
plt.xlabel("Education Field")
plt.ylabel("Count")
plt.title("Attrition by Education Field")
# Display the plot
plt.show()

#____________________________________________________________________

# 2. Department-wise Attrition
department_attrition = df1.groupby(["Department", "Attrition"]).size().reset_index(name='count')
plt.figure(figsize=(10, 6))
sns.barplot(x="Department", y="count", hue="Attrition", data=department_attrition, palette="Set2")
plt.xlabel("Department")
plt.ylabel("Count")
plt.title("Department-wise Attrition")
plt.show()

#____________________________________________________________________

# 2. (a) Monthly income by Department
avg_income_by_department = df1.groupby("Department")["MonthlyIncome"].mean()
spectral_palette = sns.color_palette("Spectral", len(avg_income_by_department))
plt.figure(figsize=(12, 6))
avg_income_by_department.plot(kind="bar", color=spectral_palette, alpha=0.7)
plt.xlabel("Department")
plt.ylabel("Average Monthly Income")
plt.title("Average Monthly Income by Department")
plt.show()

#____________________________________________________________________

# 3. Environment Satisfaction wise Attrition
# Group by Attrition and EnvironmentSatisfaction, then get the count
env_satisfaction_by_attrition = df1.groupby(["Attrition", "EnvironmentSatisfaction"]).size().reset_index(name='count')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
# Plot for Attrition=Yes
attrition_yes = env_satisfaction_by_attrition[env_satisfaction_by_attrition["Attrition"] == "Yes"]
sns.barplot(x="EnvironmentSatisfaction", y="count", data=attrition_yes, ax=ax1, palette='Reds')
ax1.set_title("Environment Satisfaction (Attrition = Yes)")
ax1.set_xlabel("Environment Satisfaction")
ax1.set_ylabel("Count")
# Plot for Attrition=No
attrition_no = env_satisfaction_by_attrition[env_satisfaction_by_attrition["Attrition"] == "No"]
sns.barplot(x="EnvironmentSatisfaction", y="count", data=attrition_no, ax=ax2, palette='Blues')
ax2.set_title("Environment Satisfaction (Attrition = No)")
ax2.set_xlabel("Environment Satisfaction")
plt.tight_layout()  
plt.show()

#____________________________________________________________________

# 4. Percentage of Attrition by Gender
gender_attrition_counts = df1.groupby(["Gender", "Attrition"]).size().reset_index(name='count')
gender_totals = df1.groupby("Gender").size().reset_index(name='total')
gender_attrition_percent = pd.merge(gender_attrition_counts, gender_totals, on="Gender")
gender_attrition_percent["percentage"] = (gender_attrition_percent["count"] / gender_attrition_percent["total"] * 100).round(2)
plt.figure(figsize=(10, 6))
sns.barplot(x="Gender", y="percentage", hue="Attrition", data=gender_attrition_percent, palette=["#3E89C3", "#FAAAB5"])
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.title("Percentage of Attrition by Gender")
plt.show()

#______________________________________________________________________

# 4. (a) Average Years at Company by Gender
plt.figure(figsize=(10, 6))
df1.groupby("Gender")["YearsAtCompany"].mean().plot(kind="bar", color=["blue", "pink"], alpha=0.7)
plt.ylabel("Average Years at Company")
plt.xlabel("Gender")
plt.title("Average Years at Company by Gender")
plt.show()
