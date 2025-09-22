# Explainations for the work are being added as comments
import sqlite3
import pandas as pd

# Current db file location
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

"""
For this task, I identify spammer by check the same contents being posted or commented more than 3 times by the same user
I use 2 separate SELECT to find the spam and combine them using UNION.
I also add a column 'type' to indicate whether the spam is from post or comment.
"""

try:
    spammer_df = pd.read_sql_query("""
    SELECT
        users.username,
        posts.content,
        'post' as type,
        COUNT(*) as occur
    FROM posts
    JOIN users on users.id = posts.user_id
    GROUP by posts.user_id, posts.content
    HAVING COUNT(*) >= 3

    UNION

    SELECT
        users.username,
        comments.content,
        'comment' as type,
        COUNT(*) as occur
    FROM comments
    JOIN users on users.id = comments.user_id
    GROUP by comments.user_id, comments.content
    HAVING COUNT(*) >= 3;
    """, conn)
    print("Spammer: ")
    print(spammer_df)
except Exception as e:
    print(f"Error: {e}")

"""
Output:
Spammer: 
        username                                            content     type  occur
0    coding_whiz  ?FREE VACATION? Tag a friend you’d take to Bal...  comment      3
1    coding_whiz  Shocking! #lol #weekend #coffee #bookstagram #...     post      3
2    coding_whiz  Top 10 gadgets of 2025 – All available here: b...     post      8
3    eco_warrior  Not gonna lie, I was skeptical at first. But a...     post      6
4    eco_warrior  Revolutionary idea! #fashionblogger #instafash...     post      3
5    eco_warrior  Wearing this hoodie in my latest reel—so many ...     post      4
6   history_buff  A lot of you asked what helped me drop 5kg in ...     post      5
7   history_buff  Best way to clean your sneakers ? snag yours h...     post      5
8   history_buff  Mood: me refreshing for likes every 30 seconds...     post      5
9   history_buff  What do you think? #thoughts #motivationmonday...     post      4
10  history_buff  You need this travel pillow in your life ? sho...     post      3
11     night_owl  ? Mega Giveaway Alert! ? Follow all accounts w...     post      8
12     night_owl  ?FLASH GIVEAWAY? Click the link in our bio to ...     post      5
13     night_owl  Find out why everyone is switching to this new...     post      4
14     night_owl  This one trick will make you $500/day from hom...     post      3
15     yoga_yogi  I couldn’t believe it! I just entered this giv...     post      5
16     yoga_yogi  Just entered this Xbox giveaway and the form w...     post      3

"""