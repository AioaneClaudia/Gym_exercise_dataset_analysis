# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("-- Start Preprocessing --")

# LOAD DATA
input_path = "data/gym_members_exercise_tracking.csv"
df = pd.read_csv(input_path)

# 1. CLEAN - pulsul maxim nu poate fi mai mic decat media pulsului ca sa nu faca predictii gresite
df = df[df['Avg_BPM'] <= df['Max_BPM']].copy()

# 2. ENCODE - transformarea datelor in valori numerice.
# LabelEncoder si One-Hot Encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Workout_Type'], prefix='Type')

for col in df.columns:
    if 'Type_' in col:
        df[col] = df[col].astype(int)

# 3. SPLIT - separarea intrebarii de raspuns.
target = "Workout_Frequency (days/week)"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SCALE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)

# 5. SAVE
X_train_final.to_csv("data/X_train.csv", index=False)
X_test_final.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print(f"Preprocessing komplett! Train: {len(X_train_final)}, Test: {len(X_test_final)}")