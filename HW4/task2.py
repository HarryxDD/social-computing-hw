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
    for idx in range(20):
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
        
        topic_display = f"{topic_name}-({topic_no})"
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
1      Entertainment & Media-(17)          0.5401       53       24.7     4.3
2      Daily Life & Reflections-(0)        0.4645       60       22.6     2.6
3      Travel & Photography-(4)            0.4042       68       16.3     2.9
4      Mental Health-(1)                   0.3817       49       19.1     3.2
5      Nature & Outdoors-(7)               0.3715       57       19.9     5.3
6      Entertainment & Media-(15)          0.3518       65       18.1     4.5
7      Climate & Environment-(19)          0.3477       78       16.3     3.2
8      Daily Life & Reflections-(18)       0.3340       62       17.4     4.8
9      Daily Life & Reflections-(3)        0.3168       43       17.7     5.8
10     Daily Life & Reflections-(12)       0.3149       87       19.2     6.1
11     Daily Life & Reflections-(5)        0.3133       53       17.5     5.1
12     Technology & Gaming-(2)             0.3072       64       18.4     4.6
13     Food & Cooking-(11)                 0.3027       68       21.7     7.0
14     Personal Growth-(13)                0.2901       51       17.0     6.3
15     Daily Life & Reflections-(9)        0.2774       42       14.3     3.3
16     Personal Growth-(14)                0.2573       76       16.1     6.5
17     Mental Health-(6)                   0.2409       98       22.3     9.6
18     Books & Reading-(10)                0.1894       92       16.6     9.2
19     Politics & News-(8)                 0.1430       84       14.1     7.8
20     Community & Social-(16)             0.1202       53       16.7     11.0

TOP 3 MOST POSITIVE TOPICS

1. Entertainment & Media
   Keywords: meme, best, new
   Average compound: 0.5401 | Median: 0.6763
   Number of posts: 53
   Positive proportion: 0.247 | Negative proportion: 0.043

2. Daily Life & Reflections
   Keywords: today, simple, art
   Average compound: 0.4645 | Median: 0.5960
   Number of posts: 60
   Positive proportion: 0.226 | Negative proportion: 0.026

3. Travel & Photography
   Keywords: new, cant, wait
   Average compound: 0.4042 | Median: 0.5168
   Number of posts: 68
   Positive proportion: 0.163 | Negative proportion: 0.029

TOP 3 MOST NEGATIVE TOPICS (Actually 'Least Positive')

1. Community & Social
   Keywords: people, damn, every
   Average compound: 0.1202 | Median: 0.4215
   Number of posts: 53
   Positive proportion: 0.167 | Negative proportion: 0.110

2. Politics & News
   Keywords: another, new, debate
   Average compound: 0.1430 | Median: 0.1230
   Number of posts: 84
   Positive proportion: 0.141 | Negative proportion: 0.078

3. Books & Reading
   Keywords: people, book, cant
   Average compound: 0.1894 | Median: 0.3902
   Number of posts: 92
   Positive proportion: 0.166 | Negative proportion: 0.092

Visualization saved as 'sentiment_visualization.png'
"""