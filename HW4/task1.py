# Explainations for the work are being added as comments
import sqlite3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from collections import Counter  # More efficient for counting
import warnings
warnings.filterwarnings('ignore')

"""
After researching different approaches for determining the optimal number of topics, I chose to use Coherence Score Optimization with grid search. There is actually different methods, Hierarchical Dirich Process is one of them, but it could lead to less interpretable and unstable results.

The coherence score (c_v) is the best metric for evaluating topic models because it measures how semantically related the words in each topic are. Higher coherence indicates more interpretable and distinct topics. This is widely used in research papers. For our dataset of 1303 posts, testing K from 5 to 20 is reasonable and fast because it covers both broad themes and specific topics. It also provides a clear, explainable methodology.

Steps:
- Extract all posts from the database using SQL
- Preprocess the text (cleaning, tokenization, lemmatization, stopword removal)
- Create a dictionary and corpus for LDA
- Test different K values (5-20) and calculate coherence score for each
- Select the K with the highest coherence score as optimal
- Train the final LDA model with optimal K
- Identify and rank the top 10 topics by number of posts
"""

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Current db file location
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove URLs/special chars, tokenize, remove stopwords, lemmatize
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag words
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters
    
    tokens = word_tokenize(text)
    
    # Remove stopwords and short tokens
    stop_words = set(stopwords.words('english'))

    """
    I add some custom stopwords that are too generic and do not contribute to topic meaning
    """
    custom_stopwords = {
        'today', 'day', 'time', 'got', 'went', 'made', 'make', 'tried', 'try', 'started', 'spent', 'new', 'old', 'still', 'finally', 'just', 'perfect', 'amazing', 'bad', 'like', 'love', 'really', 'very', 'always', 'never', 'thing', 'things', 'something', 'anything', 'people', 'actually', 'basically', 'literally', 'damn', 'now', 'then', 'ago'
    }

    stop_words.update(custom_stopwords)

    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize to reduce words to base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def load_and_preprocess_posts():
    """Load posts from database and preprocess them"""
    posts_df = pd.read_sql_query("SELECT id, content FROM posts WHERE content IS NOT NULL AND content != '';", conn)
    print(f"Loaded {len(posts_df)} posts from database")
    
    # Invoke preprocess function using apply
    posts_df['processed_tokens'] = posts_df['content'].apply(preprocess_text)
    return posts_df

def generate_topic_label(top_words):
    """
    Generate an interpretable label for a topic based on its top words.
    
    NOTE: Since LDA only produces word distributions without semantic labels, topic labels were assigned manually based on the top 10 words for each topic, following standard practice in topic modeling research. I chose this approach because automated labeling methods often lack the nuance and context that human interpretation provides. At the end, those methods are based on what humans understand from the words.
    
    The function uses keyword matching to suggest labels, but these should be reviewed and adjusted based on domain knowledge and context.
    """
    keywords_map = {
        'Food & Cooking': ['recipe', 'cooking', 'cook', 'chef', 'baking', 'meal', 'dish', 'dinner', 'lunch', 'ingredient', 'kitchen', 'delicious', 'tasty', 'flavor', 'eat', 'food'],
        'Fitness & Health': ['workout', 'fitness', 'gym', 'exercise', 'training', 'cardio', 'muscle', 'strength', 'running', 'weight', 'yoga', 'reps', 'protein', 'health', 'fit'],
        'Mental Health': ['mental', 'therapy', 'anxiety', 'depression', 'mindfulness', 'wellness', 'selfcare', 'healing', 'counseling', 'mentalhealth'],
        'Books & Reading': ['book', 'read', 'reading', 'novel', 'author', 'library', 'literature', 'chapter', 'page', 'fiction', 'story', 'bookclub', 'bookworm', 'writing', 'meme'],
        'Nature & Outdoors': ['hiking', 'trail', 'mountain', 'forest', 'wilderness', 'camping', 'wildlife', 'backpacking', 'naturelover', 'scenic', 'peak', 'outdoor', 'adventure', 'nature'],
        'Technology & Gaming': ['tech', 'game', 'gaming', 'gamer', 'console', 'gameplay', 'streamer', 'coding', 'programming', 'software', 'computer', 'digital', 'esports', 'technology'],
        'Politics & News': ['politics', 'political', 'government', 'election', 'vote', 'policy', 'legislation', 'campaign', 'senate', 'congress', 'democrat', 'republican', 'debate'],
        'DIY & Crafts': ['diy', 'craft', 'crafting', 'handmade', 'woodworking', 'sewing', 'knitting', 'pottery', 'maker', 'upcycling', 'tutorial', 'project', 'build', 'building'],
        'Travel & Photography': ['travel', 'trip', 'vacation', 'destination', 'wanderlust', 'photography', 'photographer', 'camera', 'lens', 'passport', 'tourist', 'journey', 'explore'],
        'Community & Social': ['community', 'volunteer', 'charity', 'nonprofit', 'activism', 'neighbor', 'civic', 'volunteering', 'local', 'outreach', 'fundraiser', 'together', 'support'],
        'Climate & Environment': ['climate', 'climatechange', 'environmental', 'sustainability', 'renewable', 'carbon', 'emissions', 'conservation', 'pollution', 'green', 'ecofriendly', 'environment', 'change'],
        'Fashion & Style': ['fashion', 'style', 'outfit', 'fashionista', 'trendy', 'wardrobe', 'designer', 'clothing', 'wear', 'dress', 'aesthetic', 'ootd', 'look'],
        'Art & Creativity': ['art', 'artist', 'painting', 'drawing', 'sculpture', 'gallery', 'canvas', 'illustration', 'sketch', 'artistic', 'creative', 'music', 'design'],
        'Entertainment & Media': ['movie', 'film', 'series', 'streaming', 'netflix', 'show', 'episode', 'watch', 'cinema', 'actor', 'hollywood', 'video', 'funny', 'meme', 'best'],
        'Personal Growth': ['growth', 'mindset', 'motivation', 'inspiration', 'goals', 'achievement', 'learning', 'wisdom', 'progress', 'perspective', 'transformation', 'development', 'real', 'energy'],
        'Feelings & Emotions': ['feeling', 'feel', 'felt', 'like', 'love', 'hate', 'happy', 'sad', 'mood', 'emotion'],
        'Daily Moments': ['today', 'simple', 'moment'],
        'Routine & Habits': ['day', 'classic', 'routine', 'daily', 'morning', 'coffee', 'breakfast'],
        'Current State': ['still', 'now', 'currently'],
        'Quality & Perfection': ['perfect', 'great', 'awesome', 'amazing'],
        'Time Spent': ['time', 'spent', 'hour', 'week', 'weekend', 'year', 'ago'],
        'Trying Things': ['tried', 'try', 'attempt', 'testing'],
        'Getting Things': ['got', 'bought', 'received', 'latest'],
        'Completing Things': ['finished', 'completed', 'done', 'finally'],
        'New Experiences': ['new', 'first', 'discover', 'fresh', 'another', 'recent'],
        'Social Observations': ['people', 'everyone', 'someone', 'anyone', 'damn', 'guy', 'person', 'friend'],
        'Life Philosophy': ['life', 'never', 'always', 'think', 'thought'],
        'Understanding': ['knew', 'know', 'understand', 'realize'],
    }
 
    # Count matches for each category with better scoring
    category_scores = []
    
    """
    Here I'm using a weighted scoring system to prioritize top words more heavily. The top 3 words get a weight of 3, the next 2 words get a weight of 2, and the remaining words get a weight of 1. This way, if a category matches with the most significant words of the topic, it will score higher and be more likely to be selected as the label
    """
    for category, keywords in keywords_map.items():
        score = 0
        for i, word in enumerate(top_words[:10]):
            if word in keywords:
                weight = 3 if i < 3 else (2 if i < 5 else 1)  # Higher weight for top words
                score += weight
        if score > 0:
            category_scores.append((category, score))
    
    # Get best match
    if category_scores:
        # Sort by score and avoid duplicates by checking if label was already used
        category_scores.sort(key=lambda x: x[1], reverse=True)
        best_match = category_scores[0][0]
    else:
        # Fallback: create descriptive label from top 3 words
        best_match = f"Topic: {', '.join(top_words[:3])}"
    
    return best_match

def create_dictionary_and_corpus(documents):
    """Create dictionary and corpus (bag of words) for LDA"""
    dictionary = corpora.Dictionary(documents)
    
    # Filter extremes: remove words that appear in <5 documents or >50% of documents
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    print(f"Dictionary created with {len(dictionary)} terms")
    
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    return dictionary, corpus

def find_optimal_k(corpus, dictionary, documents, k_range=range(2, 21)):
    """
    Find the optimal number of topics K by testing different values and comparing coherence scores.
    
    Coherence score measures how interpretable the topics are. Higher coherence means the words in each topic are more semantically related, making topics more meaningful and distinct.

    We use c_v coherence as it correlates well with human topic interpretability judgments.

    c_v is based on a sliding window, a one-set segmentation of the top words, and an indirect confirmation measure that uses normalized pointwise mutual information and the cosine similarity.
    """
    print("\nFinding optimal K by testing different numbers of topics...")
    print(f"Testing K values from {min(k_range)} to {max(k_range)}")
    print(f"{'K':<5} {'Coherence Score':<20}")
    print("-" * 25)
    
    coherence_scores = []
    
    for k in k_range:
        # Train LDA model with k topics
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=10,  # Reduced for faster testing
            iterations=200,
            alpha='auto',
            eta='auto',
            per_word_topics=True
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        coherence_scores.append((k, coherence))
        print(f"{k:<5} {coherence:<20.4f}")
    
    # Find K with highest coherence
    optimal_k, best_coherence = max(coherence_scores, key=lambda x: x[1])
    print(f"\nOptimal K = {optimal_k} (Coherence: {best_coherence:.4f})")
    
    return optimal_k, coherence_scores

def train_lda_model(corpus, dictionary, num_topics):
    """Train final LDA model with optimal number of topics"""
    print(f"\nTraining final LDA model with K={num_topics} topics...")
    
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=15,  # More passes for final model
        iterations=400,
        alpha='auto',  # Auto-learn document-topic density
        eta='auto',  # Auto-learn topic-word density
        per_word_topics=True
    )
    
    print("Model training completed")
    return lda_model

def evaluate_model(lda_model, corpus, dictionary, documents):
    """
    Evaluate final model quality using coherence score and perplexity.
    
    Coherence measures topic interpretability (higher is better).
    Perplexity measures how well the model predicts the data (lower is better).
    """
    # Calculate coherence score
    coherence_model = CoherenceModel(
        model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    
    # Calculate perplexity
    perplexity = lda_model.log_perplexity(corpus)
    
    print(f"Final Model Coherence Score: {coherence_score:.4f} (higher is better)")
    print(f"Final Model Perplexity: {perplexity:.4f} (lower is better)")
    
    return coherence_score, perplexity

def extract_and_display_topics(lda_model, num_topics, num_words=10):
    """Extract and display all topics with their top words"""
    topics_data = []
    
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        words_weights = []
        for item in topic.split(' + '):
            weight, word = item.split('*')
            word = word.strip('"')
            weight = float(weight)
            words_weights.append((word, weight))
        
        top_words = [word for word, weight in words_weights]
        
        # Generate interpretable topic label based on top words
        topic_label = generate_topic_label(top_words)
        
        topics_data.append({
            'topic_id': idx, 
            'label': topic_label,
            'words': words_weights, 
            'top_words': top_words
        })
    
    # Fix duplicate labels by adding distinguishing words
    # This is because I detected some topics had same labels in the output
    label_counts = Counter(t['label'] for t in topics_data)
    for topic in topics_data:
        if label_counts[topic['label']] > 1:
            # Add top 2 distinctive words to make label unique
            topic['label'] = f"{topic['label']}: {', '.join(topic['top_words'][:2])}"
    
    return topics_data

def analyze_topic_distribution(lda_model, corpus, num_topics, topics_data):
    """
    Analyze topic distribution and identify the TOP 10 topics with the most posts.
    
    For this, I used Counter instead of dict + manual counting for better performance. Counter.most_common() uses a heap internally, which is more efficient than sorting all items when you only need the top K.
    """
    print(f"\n{'='*80}")
    print("TOPIC DISTRIBUTION (All Topics)")
    print(f"{'='*80}\n")
    
    topic_counter = Counter()
    for doc_topics in lda_model.get_document_topics(corpus):
        if doc_topics:
            dominant_topic = max(doc_topics, key=lambda x: x[1])[0]
            topic_counter[dominant_topic] += 1
    
    total = len(corpus)
    for topic_id in range(num_topics):
        count = topic_counter.get(topic_id, 0)  # Default to 0 if topic has no posts
        pct = (count / total) * 100
        label = topics_data[topic_id]['label']
        print(f"Topic {topic_id + 1} ({label}): {count} posts ({pct:.1f}%)")
    
    # Identify TOP 10 topics by post count
    print(f"\n{'='*80}")
    print("TOP 10 TOPICS WITH MOST POSTS (Answer to Exercise 4.1)")
    print(f"{'='*80}\n")
    
    # most_common(10) uses heap internally - O(n log k)
    top_10_topics = topic_counter.most_common(10)
    
    print(f"{'Rank':<6} {'Topic':<10} {'Topic Name':<30} {'Posts':<10} {'%':<8}")
    print("-" * 70)
    for rank, (topic_id, count) in enumerate(top_10_topics, 1):
        pct = (count / total) * 100
        label = topics_data[topic_id]['label']
        print(f"{rank:<6} Topic {topic_id + 1:<3} {label:<30} {count:<10} {pct:>6.1f}%")
    
    return topic_counter, top_10_topics

def main():
    # Load and preprocess posts
    posts_df = load_and_preprocess_posts()
    documents = posts_df['processed_tokens'].tolist()
    
    # Create dictionary and corpus
    dictionary, corpus = create_dictionary_and_corpus(documents)
    
    # Find optimal K by comparing coherence scores
    # I chose range 5-20 based on dataset size and diversity
    # It can cover both broad themes and specific topics
    optimal_k, coherence_scores = find_optimal_k(corpus, dictionary, documents, k_range=range(5, 21))

    # Train final LDA model with optimal K
    lda_model = train_lda_model(corpus, dictionary, num_topics=optimal_k)

    # Save model and dictionary for use in task 4.2
    lda_model.save('lda_model_k20.model')
    dictionary.save('lda_dictionary.dict')

    # Evaluate final model
    coherence_score, perplexity = evaluate_model(lda_model, corpus, dictionary, documents)

    # Extract and display all topics
    topics_data = extract_and_display_topics(lda_model, num_topics=optimal_k, num_words=10)

    # Analyze distribution and identify TOP 10 topics with most posts
    analyze_topic_distribution(lda_model, corpus, num_topics=optimal_k, topics_data=topics_data)
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

"""
Loaded 1303 posts from database
Dictionary created with 685 terms

Finding optimal K by testing different numbers of topics...
Testing K values from 5 to 20
K     Coherence Score     
-------------------------
5     0.3700
6     0.3759
7     0.4289
8     0.4130
9     0.3971
10    0.4312
11    0.3854
12    0.4132
13    0.4089
14    0.4230
15    0.4187
16    0.3891
17    0.3877
18    0.4053
19    0.4114
20    0.3980

# This optimal K could be affected by the change of custom stopwords or preprocessing steps
Optimal K = 10 (Coherence: 0.4312)

Training final LDA model with K=10 topics...
Model training completed
Final Model Coherence Score: 0.4152 (higher is better)
Final Model Perplexity: -6.4902 (lower is better)

================================================================================
TOPIC DISTRIBUTION (All Topics)
================================================================================

Topic 1 (Understanding): 131 posts (10.1%)
Topic 2 (Feelings & Emotions): 146 posts (11.2%)
Topic 3 (Life Philosophy): 115 posts (8.8%)
Topic 4 (Personal Growth): 109 posts (8.4%)
Topic 5 (Fitness & Health): 154 posts (11.8%)
Topic 6 (Books & Reading): 120 posts (9.2%)
Topic 7 (Entertainment & Media): 102 posts (7.8%)
Topic 8 (DIY & Crafts): 112 posts (8.6%)
Topic 9 (Politics & News): 145 posts (11.1%)
Topic 10 (Nature & Outdoors): 169 posts (13.0%)

================================================================================
TOP 10 TOPICS WITH MOST POSTS (Answer to Exercise 4.1)
================================================================================

Rank   Topic      Topic Name                     Posts      %
----------------------------------------------------------------------
1      Topic 10  Nature & Outdoors              169          13.0%
2      Topic 5   Fitness & Health               154          11.8%
3      Topic 2   Feelings & Emotions            146          11.2%
4      Topic 9   Politics & News                145          11.1%
5      Topic 1   Understanding                  131          10.1%
6      Topic 6   Books & Reading                120           9.2%
7      Topic 3   Life Philosophy                115           8.8%
8      Topic 8   DIY & Crafts                   112           8.6%
9      Topic 4   Personal Growth                109           8.4%
10     Topic 7   Entertainment & Media          102           7.8%
"""