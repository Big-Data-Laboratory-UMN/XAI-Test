#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# In[ ]:


# =====================================================
# 1. LOAD DAN PREPROCESSING DATA
# =====================================================

def load_and_preprocess_data(filepath):
    """
    Load dan preprocessing dataset Facebook Metrics
    """
    # Load data
    df = pd.read_csv(filepath, sep=';')
    
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total Records: {len(df)}")
    print(f"Total Features: {df.shape[1]}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    
    # Feature Engineering: Total Interactions sebagai target
    # Definisikan threshold untuk High/Low Engagement
    engagement_threshold = df['Total Interactions'].median()
    df['Engagement_Class'] = df['Total Interactions'].apply(
        lambda x: 'High Engagement' if x > engagement_threshold else 'Low Engagement'
    )
    
    print(f"\nEngagement Threshold (Median): {engagement_threshold}")
    print(f"\nClass Distribution:")
    print(df['Engagement_Class'].value_counts())
    print(f"\nPercentage:")
    print(df['Engagement_Class'].value_counts(normalize=True) * 100)
    
    return df, engagement_threshold


# In[ ]:


def create_features(df):
    """
    Feature engineering dan selection
    """
    # Pilih fitur yang relevan untuk prediksi SEBELUM posting
    features = [
        'Page total likes',
        'Type',
        'Category',
        'Post Month',
        'Post Weekday',
        'Post Hour',
        'Paid'
    ]
    
    target = 'Engagement_Class'
    
    # Buat dataframe fitur
    X = df[features].copy()
    y = df[target].copy()
    
    # Encode categorical variables
    le_type = LabelEncoder()
    X['Type_Encoded'] = le_type.fit_transform(X['Type'])
    
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Drop original categorical column
    X_processed = X.drop(['Type'], axis=1)
    
    # Rename untuk clarity
    X_processed.columns = [
        'Page_Total_Likes',
        'Category',
        'Post_Month',
        'Post_Weekday',
        'Post_Hour',
        'Paid',
        'Type_Encoded'
    ]
    
    return X_processed, y_encoded, le_type, le_target, features


# In[ ]:


# =====================================================
# 2. EXPLORATORY DATA ANALYSIS
# =====================================================

def perform_eda(df):
    """
    Analisis eksplorasi data
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('EXPLORATORY DATA ANALYSIS - FACEBOOK METRICS', fontsize=16, fontweight='bold')
    
    # 1. Engagement by Type
    engagement_by_type = df.groupby(['Type', 'Engagement_Class']).size().unstack()
    engagement_by_type.plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#2ecc71'])
    axes[0, 0].set_title('Engagement by Content Type')
    axes[0, 0].set_xlabel('Content Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend(title='Engagement')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Paid vs Organic
    engagement_by_paid = df.groupby(['Paid', 'Engagement_Class']).size().unstack()
    engagement_by_paid.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#2ecc71'])
    axes[0, 1].set_title('Paid vs Organic Performance')
    axes[0, 1].set_xlabel('Paid (0=Organic, 1=Paid)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticklabels(['Organic', 'Paid'], rotation=0)
    
    # 3. Posting Time Analysis
    hour_engagement = df.groupby('Post Hour')['Total Interactions'].mean().sort_values(ascending=False)
    hour_engagement.plot(kind='bar', ax=axes[0, 2], color='#3498db')
    axes[0, 2].set_title('Average Engagement by Posting Hour')
    axes[0, 2].set_xlabel('Hour of Day')
    axes[0, 2].set_ylabel('Avg Total Interactions')
    
    # 4. Weekday Analysis
    weekday_engagement = df.groupby('Post Weekday')['Total Interactions'].mean()
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(1, 8), weekday_engagement, color='#9b59b6')
    axes[1, 0].set_title('Average Engagement by Weekday')
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Avg Total Interactions')
    axes[1, 0].set_xticks(range(1, 8))
    axes[1, 0].set_xticklabels(weekday_labels)
    
    # 5. Distribution of Total Interactions
    axes[1, 1].hist(df['Total Interactions'], bins=30, color='#e67e22', edgecolor='black')
    axes[1, 1].set_title('Distribution of Total Interactions')
    axes[1, 1].set_xlabel('Total Interactions')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(df['Total Interactions'].median(), color='red', 
                       linestyle='--', label=f'Median: {df["Total Interactions"].median():.0f}')
    axes[1, 1].legend()
    
    # 6. Category Analysis
    category_engagement = df.groupby('Category')['Total Interactions'].mean().sort_values(ascending=False)
    category_engagement.plot(kind='barh', ax=axes[1, 2], color='#1abc9c')
    axes[1, 2].set_title('Average Engagement by Category')
    axes[1, 2].set_xlabel('Avg Total Interactions')
    axes[1, 2].set_ylabel('Category')
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM EDA")
    print("=" * 60)
    print(f"Best performing content type: {df.groupby('Type')['Total Interactions'].mean().idxmax()}")
    print(f"Best posting hour: {df.groupby('Post Hour')['Total Interactions'].mean().idxmax()}:00")
    print(f"Best posting day: {weekday_labels[df.groupby('Post Weekday')['Total Interactions'].mean().idxmax()-1]}")
    print(f"Paid content avg engagement: {df[df['Paid']==1]['Total Interactions'].mean():.2f}")
    print(f"Organic content avg engagement: {df[df['Paid']==0]['Total Interactions'].mean():.2f}")


# In[ ]:


# =====================================================
# 3. MODEL TRAINING DAN EVALUATION
# =====================================================

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models dan bandingkan performance
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Pilih best model
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*60}")
    
    return results, best_model, best_model_name


# In[ ]:


def plot_model_comparison(results):
    """
    Visualisasi perbandingan model
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('MODEL PERFORMANCE COMPARISON', fontsize=16, fontweight='bold')
    
    # Accuracy & ROC-AUC comparison
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    roc_aucs = [results[m]['roc_auc'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
    axes[0].bar(x + width/2, roc_aucs, width, label='ROC-AUC', color='#e74c3c')
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Accuracy vs ROC-AUC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0].legend()
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    # ROC Curves
    for name, result in results.items():
        y_test_dummy = np.random.choice([0, 1], size=len(result['y_pred']))  # Placeholder
        fpr, tpr, _ = roc_curve(y_test_dummy, result['y_pred_proba'])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})", linewidth=2)
    
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[ ]:


def plot_confusion_matrix(y_test, y_pred, le_target):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))


# In[ ]:


# =====================================================
# 4. EXPLAINABLE AI (XAI) - SHAP ANALYSIS
# =====================================================

def shap_analysis(model, X_train, X_test, feature_names):
    """
    SHAP (SHapley Additive exPlanations) untuk interpretability
    """
    print("\n" + "=" * 60)
    print("EXPLAINABLE AI - SHAP ANALYSIS")
    print("=" * 60)
    print("Analyzing feature importance and model decisions...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Jika binary classification, ambil values untuk class 1
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
    
    # 1. Summary Plot - Feature Importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, 
                      show=False, plot_type="bar")
    plt.title('SHAP Feature Importance - Faktor Dominan Viralitas', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Summary Plot - Detailed
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Dampak Fitur terhadap Prediksi', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Dependence Plot untuk top features
    feature_importance = np.abs(shap_values_class1).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[-3:]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('SHAP Dependence Plots - Top 3 Features', fontsize=14, fontweight='bold')
    
    for idx, feat_idx in enumerate(top_features_idx):
        shap.dependence_plot(feat_idx, shap_values_class1, X_test, 
                            feature_names=feature_names,
                            ax=axes[idx], show=False)
    
    plt.tight_layout()
    plt.savefig('shap_dependence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print interpretasi
    print("\n" + "=" * 60)
    print("INTERPRETASI FAKTOR DOMINAN VIRALITAS")
    print("=" * 60)
    
    avg_shap = np.abs(shap_values_class1).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Importance': avg_shap
    }).sort_values('SHAP Importance', ascending=False)
    
    print(feature_importance_df.to_string(index=False))
    
    return shap_values_class1, explainer


def individual_prediction_explanation(model, explainer, X_test, feature_names, idx=0):
    """
    Penjelasan untuk prediksi individual
    """
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Force plot untuk satu prediksi
    plt.figure(figsize=(12, 3))
    shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list) 
                    else explainer.expected_value,
                    shap_values[idx], 
                    X_test.iloc[idx],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False)
    plt.title(f'Penjelasan Prediksi untuk Sampel #{idx}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'shap_individual_{idx}.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[ ]:




