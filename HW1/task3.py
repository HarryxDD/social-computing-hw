# Explainations for the work are being added as comments
import sqlite3
import pandas as pd

# Current db file location
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

"""
To find top 5 influencers, I count the number of reactions and comments on each user's posts.
First, I JOIN the posts table with the users table to get the username (the author).
Then, I LEFT JOIN the reactions and comments tables to count the number of reactions and comments for each posts.
Finally, I group the results by username and order them by the total number of reactions and comments in descending order, limiting the results to the top 5.

By using DISTINCT in the COUNT, I ensure that each reaction and is counted only once, because when joining multiple tables, there can be duplicate rows for the same reaction and comment, resulting in same count value for these columns.
"""

try:
    influencer_df = pd.read_sql_query("""
    SELECT 
	    users.id, 
        users.username, 
        COUNT(DISTINCT reactions.id) as Reactions, 
        COUNT(DISTINCT comments.id) AS Comments 
    FROM posts
    JOIN users on users.id = posts.user_id
    JOIN reactions on posts.id = reactions.post_id
    JOIN comments ON posts.id = comments.post_id
    GROUP by users.username
    ORDER BY (COUNT(DISTINCT reactions.id) + COUNT(DISTINCT comments.id)) DESC
    LIMIT 5;
    """, conn)
    print("Top 5 influencers: ")
    print(influencer_df)
except Exception as e:
    print(f"Error: {e}")

"""
Output:
Top 5 influencers: 
   id      username  Reactions  Comments
0  54    WinterWolf        267       179
1  65   PinkPanther        234       152
2  94     PinkPetal        246       137
3  81  GoldenDreams        217       149
4  30     WildHorse        196       157

"""