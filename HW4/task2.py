# Explainations for the work are being added as comments
import sqlite3
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
# Import preprocess_text and generate_topic_label from task1
from task1 import preprocess_text, generate_topic_label
from gensim.models import LdaModel
from gensim import corpora

warnings.filterwarnings('ignore')

"""
For this analysis, I will:
- Analyze sentiment of all posts and comments using VADER
- Calculate overall platform sentiment (compound scores)
- Link posts to topics from Exercise 4.1 (import it) using the same LDA model
- Compare sentiment across different topics
- Provide insights on which topics have more positive/negative sentiment
"""

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Database connection
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

sia = SentimentIntensityAnalyzer()

def analyze_sentiment_batch(df, content_column='content'):
    """
    Analyze sentiment using VADER and return all scores.
    
    VADER returns sentiment_dict, which includes:
    - compound: Overall sentiment (-1 most negative to +1 most positive)
    - pos: Positive sentiment proportion
    - neu: Neutral sentiment proportion  
    - neg: Negative sentiment proportion
    
    The compound score interpretation:
    - >= 0.05: Positive
    - <= -0.05: Negative
    - Between -0.05 and 0.05: Neutral

    I choose these points because they are standard thresholds
    """

    # Generic function for both posts and comments
    sentiment_results = df[content_column].apply(lambda x: sia.polarity_scores(x))
    
    df['compound'] = sentiment_results.apply(lambda x: x['compound'])
    df['positive'] = sentiment_results.apply(lambda x: x['pos'])
    df['neutral'] = sentiment_results.apply(lambda x: x['neu'])
    df['negative'] = sentiment_results.apply(lambda x: x['neg'])
    df['sentiment_category'] = df['compound'].apply(categorize_sentiment)
    
    return df

def categorize_sentiment(compound_score):
    """Categorize sentiment based on compound score"""
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_posts_sentiment():
    # Load all posts that have content
    posts_df = pd.read_sql_query("""
        SELECT id, user_id, content, created_at 
        FROM posts 
        WHERE content IS NOT NULL AND content != ''
    """, conn)
    
    print(f"Analyzing {len(posts_df)} posts...")
    
    return analyze_sentiment_batch(posts_df)

def analyze_comments_sentiment():
    # Load all comments that have content
    comments_df = pd.read_sql_query("""
        SELECT id, post_id, user_id, content, created_at 
        FROM comments 
        WHERE content IS NOT NULL AND content != ''
    """, conn)
    
    print(f"Analyzing {len(comments_df)} comments...")
    
    return analyze_sentiment_batch(comments_df)

def display_overall_sentiment(posts_df, comments_df):
    # Posts sentiment summary
    print("\n--- POSTS SENTIMENT ---")
    print(f"Average compound score: {posts_df['compound'].mean():.4f}")
    print(f"Median compound score: {posts_df['compound'].median():.4f}")
    print(f"Std deviation: {posts_df['compound'].std():.4f}")
    
    print(f"\nSentiment distribution:")
    sentiment_counts = posts_df['sentiment_category'].value_counts()
    for category in ['Positive', 'Neutral', 'Negative']:
        count = sentiment_counts.get(category, 0)
        pct = (count / len(posts_df)) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    # Comments sentiment summary
    print("\n--- COMMENTS SENTIMENT ---")
    print(f"Average compound score: {comments_df['compound'].mean():.4f}")
    print(f"Median compound score: {comments_df['compound'].median():.4f}")
    print(f"Std deviation: {comments_df['compound'].std():.4f}")
    
    print(f"\nSentiment distribution:")
    sentiment_counts = comments_df['sentiment_category'].value_counts()
    for category in ['Positive', 'Neutral', 'Negative']:
        count = sentiment_counts.get(category, 0)
        pct = (count / len(comments_df)) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    # Overall platform tone
    print("\n--- OVERALL PLATFORM TONE ---")
    """
    I calculate the overall platform tone by averaging the compound scores
    from both posts and comments
    """
    all_compound = pd.concat([posts_df['compound'], comments_df['compound']])
    avg_compound = all_compound.mean()
    
    if avg_compound >= 0.05:
        tone = "POSITIVE"
    elif avg_compound <= -0.05:
        tone = "NEGATIVE"
    else:
        tone = "NEUTRAL"
    
    print(f"Average compound score (all content): {avg_compound:.4f}")
    print(f"Overall platform tone: {tone}")
    
    if avg_compound > 0:
        print(f"\nThe platform has a slightly positive tone")
    else:
        print(f"\nThe platform has a neutral to slightly negative tone.")

def assign_topics_to_posts(posts_df):
    """
    Assign topics to posts using the LDA model from Exercise 4.1.
    This reuses the preprocessing and LDA model to ensure consistency.
    """
    print("--- Assigning topics to posts using LDA model ---")
    
    print("\nPreprocessing posts...")
    posts_df['processed_tokens'] = posts_df['content'].apply(preprocess_text)
    documents = posts_df['processed_tokens'].tolist()

    print("Loading LDA model and dictionary from disk...")
    lda_model = LdaModel.load('lda_model_k20.model')
    dictionary = corpora.Dictionary.load('lda_dictionary.dict')

    print("Creating corpus using loaded dictionary...")
    corpus = [dictionary.doc2bow(tokens) for tokens in documents]
    
    # Assign dominant topic to each post
    print("Assigning topics to posts...")
    topics = []
    for doc_topics in lda_model.get_document_topics(corpus):
        if doc_topics:
            dominant_topic = max(doc_topics, key=lambda x: x[1])[0]
            topics.append(dominant_topic)
        else:
            topics.append(-1)  # No topic assigned
    
    posts_df['topic_id'] = topics
    
    topic_labels = {}
    topic_keywords = {}  # Store keywords for display
    for idx in range(10):
        topic_words = lda_model.show_topic(idx, topn=10)
        words = [word for word, _ in topic_words]
        topic_keywords[idx] = ', '.join(words[:3])  # Keep top 3 for reference
        
        # Use the same labeling logic from task1
        label = generate_topic_label(words)
        topic_labels[idx] = label
    
    # Assign topic labels and keywords
    posts_df['topic_label'] = posts_df['topic_id'].apply(lambda x: topic_labels.get(x, 'Unknown'))
    posts_df['topic_keywords'] = posts_df['topic_id'].apply(lambda x: topic_keywords.get(x, ''))
    
    print(f"Successfully assigned topics to {len(posts_df[posts_df['topic_id'] != -1])} posts")
    
    return posts_df, topic_labels, topic_keywords

def analyze_sentiment_by_topic(posts_df, topic_labels, topic_keywords):
    # Filter out posts without topics
    posts_with_topics = posts_df[posts_df['topic_id'] != -1].copy()
    
    # Group by topic and calculate sentiment statistics
    topic_sentiment = posts_with_topics.groupby('topic_id').agg({
        'compound': ['mean', 'median', 'std', 'count'],
        'positive': 'mean',
        'negative': 'mean',
        'neutral': 'mean'
    }).round(4)
    
    # Flatten column names
    topic_sentiment.columns = ['_'.join(col).strip() for col in topic_sentiment.columns.values]
    topic_sentiment = topic_sentiment.reset_index()
    
    # Add topic labels and keywords
    topic_sentiment['topic_name'] = topic_sentiment['topic_id'].apply(lambda x: topic_labels.get(x, 'Unknown'))
    topic_sentiment['keywords'] = topic_sentiment['topic_id'].apply(lambda x: topic_keywords.get(x, ''))
    
    # Sort by average compound score
    topic_sentiment = topic_sentiment.sort_values('compound_mean', ascending=False)
    
    print(f"\nSentiment Summary by Topic (sorted by average compound score):")
    print(f"{'Rank':<6} {'Topic Name':<35} {'Avg Score':<12} {'Posts':<8} {'Pos%':<8} {'Neg%':<8}")
    print("-"*80)
    
    for rank, (_, row) in enumerate(topic_sentiment.iterrows(), 1):
        topic_no = row['topic_id']
        topic_name = row['topic_name']
        avg_score = row['compound_mean']
        post_count = int(row['compound_count'])
        pos_pct = row['positive_mean'] * 100
        neg_pct = row['negative_mean'] * 100
        
        topic_display = f"{topic_name}"
        print(f"{rank:<6} {topic_display:<35} {avg_score:<12.4f} {post_count:<8} {pos_pct:<8.1f} {neg_pct:<8.1f}")
    
    # Detailed analysis for top 3 most positive and negative topics
    print("\nTOP 3 MOST POSITIVE TOPICS")
    
    top_positive = topic_sentiment.head(3)
    for idx, (_, row) in enumerate(top_positive.iterrows(), 1):
        topic_name = row['topic_name']
        keywords = row['keywords']
        avg = row['compound_mean']
        median = row['compound_median']
        posts = int(row['compound_count'])
        pos = row['positive_mean']
        neg = row['negative_mean']
        
        print(f"\n{idx}. {topic_name}")
        print(f"   Keywords: {keywords}")
        print(f"   Average compound: {avg:.4f} | Median: {median:.4f}")
        print(f"   Number of posts: {posts}")
        print(f"   Positive proportion: {pos:.3f} | Negative proportion: {neg:.3f}")
    
    print("\nTOP 3 MOST NEGATIVE TOPICS (Actually 'Least Positive')")
    
    top_negative = topic_sentiment.tail(3).iloc[::-1]  # Reverse to show most negative first
    for idx, (_, row) in enumerate(top_negative.iterrows(), 1):
        topic_name = row['topic_name']
        keywords = row['keywords']
        avg = row['compound_mean']
        median = row['compound_median']
        posts = int(row['compound_count'])
        pos = row['positive_mean']
        neg = row['negative_mean']
        
        print(f"\n{idx}. {topic_name}")
        print(f"   Keywords: {keywords}")
        print(f"   Average compound: {avg:.4f} | Median: {median:.4f}")
        print(f"   Number of posts: {posts}")
        print(f"   Positive proportion: {pos:.3f} | Negative proportion: {neg:.3f}")
    
    return topic_sentiment

"""
I create this visualization function to easily see sentiment distribution
The first plot shows average sentiment by topic
The second plot compares sentiment distribution between posts and comments
"""
def visualize_sentiment_analysis(posts_df, comments_df, topic_sentiment):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Sentiment by Topic Plot
    plot_data = topic_sentiment.sort_values('compound_mean', ascending=True)
    colors = ['#e74c3c' if s < 0.2 else '#f39c12' if s < 0.35 else '#27ae60' 
              for s in plot_data['compound_mean']]
    
    axes[0].barh(range(len(plot_data)), plot_data['compound_mean'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(plot_data)))
    axes[0].set_yticklabels([f"{row['topic_name'][:25]}" for _, row in plot_data.iterrows()], fontsize=9)
    axes[0].set_xlabel('Average Sentiment Score', fontweight='bold')
    axes[0].set_title('Sentiment by Topic', fontsize=13, fontweight='bold')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Posts vs Comments Distribution Plot
    categories = ['Positive', 'Neutral', 'Negative']
    posts_vals = [posts_df['sentiment_category'].value_counts().get(c, 0) for c in categories]
    comments_vals = [comments_df['sentiment_category'].value_counts().get(c, 0) for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    axes[1].bar(x - width/2, posts_vals, width, label='Posts', color='#3498db', alpha=0.8)
    axes[1].bar(x + width/2, comments_vals, width, label='Comments', color='#e74c3c', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_title('Sentiment Distribution: Posts vs Comments', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentiment_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'sentiment_visualization.png'")
    plt.close()

def main():
    # Step 1: Analyze sentiment of posts
    posts_df = analyze_posts_sentiment()
    
    # Step 2: Analyze sentiment of comments
    comments_df = analyze_comments_sentiment()
    
    # Step 3: Display overall platform sentiment
    display_overall_sentiment(posts_df, comments_df)
    
    # Step 4: Assign topics to posts (using LDA from Exercise 4.1)
    posts_df, topic_labels, topic_keywords = assign_topics_to_posts(posts_df)
    
    # Step 5: Analyze sentiment variation across topics
    topic_sentiment = analyze_sentiment_by_topic(posts_df, topic_labels, topic_keywords)

    # Step 6: Visualize sentiment analysis
    visualize_sentiment_analysis(posts_df, comments_df, topic_sentiment)
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

"""
Analyzing 1303 posts...
Analyzing 5804 comments...

--- POSTS SENTIMENT ---
Average compound score: 0.3053
Median compound score: 0.4404
Std deviation: 0.4780

Sentiment distribution:
  Positive: 857 (65.8%)
  Neutral: 191 (14.7%)
  Negative: 255 (19.6%)

--- COMMENTS SENTIMENT ---
Average compound score: 0.4324
Median compound score: 0.5983
Std deviation: 0.4836

Sentiment distribution:
  Positive: 4446 (76.6%)
  Neutral: 339 (5.8%)
  Negative: 1019 (17.6%)

--- OVERALL PLATFORM TONE ---
Average compound score (all content): 0.4091
Overall platform tone: POSITIVE

The platform has a slightly positive tone
--- Assigning topics to posts using LDA model ---

Preprocessing posts...
Loading LDA model and dictionary from disk...
Creating corpus using loaded dictionary...
Assigning topics to posts...
Successfully assigned topics to 1303 posts

Sentiment Summary by Topic (sorted by average compound score):
Rank   Topic Name                          Avg Score    Posts    Pos%     Neg%
--------------------------------------------------------------------------------
1      Life Philosophy                     0.4078       117      19.8     3.4
2      Fitness & Health                    0.3930       154      19.1     3.8
3      Entertainment & Media               0.3766       101      20.7     4.5
4      Nature & Outdoors                   0.3652       169      18.2     3.9
5      Books & Reading                     0.3546       119      19.3     5.0
6      Feelings & Emotions                 0.3234       146      19.1     5.6
7      Personal Growth                     0.2984       110      20.0     6.6
8      Politics & News                     0.1833       145      16.1     8.1
9      DIY & Crafts                        0.1806       112      15.3     8.3
10     Understanding                       0.1600       130      16.1     10.3

TOP 3 MOST POSITIVE TOPICS

1. Life Philosophy
   Keywords: life, good, need
   Average compound: 0.4078 | Median: 0.5719
   Number of posts: 117
   Positive proportion: 0.198 | Negative proportion: 0.034

2. Fitness & Health
   Keywords: see, health, mental
   Average compound: 0.3930 | Median: 0.5106
   Number of posts: 154
   Positive proportion: 0.191 | Negative proportion: 0.038

3. Entertainment & Media
   Keywords: cant, coffee, best
   Average compound: 0.3766 | Median: 0.5145
   Number of posts: 101
   Positive proportion: 0.207 | Negative proportion: 0.045

TOP 3 MOST NEGATIVE TOPICS (Actually 'Least Positive')

1. Understanding
   Keywords: kid, could, knew
   Average compound: 0.1600 | Median: 0.1755
   Number of posts: 130
   Positive proportion: 0.161 | Negative proportion: 0.103

2. DIY & Crafts
   Keywords: diy, project, feel
   Average compound: 0.1806 | Median: 0.2490
   Number of posts: 112
   Positive proportion: 0.153 | Negative proportion: 0.083

3. Politics & News
   Keywords: another, anyone, else
   Average compound: 0.1833 | Median: 0.2023
   Number of posts: 145
   Positive proportion: 0.161 | Negative proportion: 0.081

Visualization saved as 'sentiment_visualization.png'
"""