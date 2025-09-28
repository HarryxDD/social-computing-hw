# Explainations for the work are being added as comments
import sqlite3
import pandas as pd

# Current db file location
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

"""
For this task, I define engagement as the total number of comments and reactions exchanged between two users on each other's posts. This means I count all individual comments and reactions that flow in both directions between a user pair.

First, I create the CTE all_engagements to gather all comments and reactions between users, ensuring that self-engagements are excluded by using WHERE (c|r).user_id != p.user_id.

The second CTE user_pairs aggregates the total engagement between each pair of users.
For example, if User A commented 2 times and reacted 3 times to User B's posts, the total engagement from User A to User B would be 5.

The third CTE mutual_engagement combines the engagements from both users in each pair to get the total mutual engagement. I joined the user_pairs table with itself to achieve this. I avoid double counting by ensuring that I only consider pairs where action_owner < post_owner (or action_owner > post_owner no matter), so each user pair appears only once in the final results regardless of who initiated more engagement.

"""

try:
    connections = pd.read_sql_query(f"""
    WITH all_engagements AS (
    SELECT 
  	    c.user_id AS action_owner,
  	    p.user_id AS post_owner,
  	    'comment' AS type,
  	    count(*) AS quantity
    FROM comments c
    JOIN posts p ON p.id = c.post_id
    WHERE c.user_id != p.user_id
    GROUP BY c.user_id, p.user_id
  
    UNION ALL
  
    SELECT 
  	    r.user_id AS action_owner,
  	    p.user_id AS post_owner,
  	    'reaction' AS type,
  	    count(*) AS quantity
    FROM reactions r
    JOIN posts p ON p.id = r.post_id
    WHERE r.user_id != p.user_id
    GROUP BY r.user_id, p.user_id
    ),
    user_pairs as (
    SELECT
  	    action_owner,
  	    post_owner,
  	    SUM(quantity) AS total_engagement
    FROM all_engagements
    GROUP BY action_owner, post_owner
    ),
    mutual_engagement AS (
    SELECT 
        CASE WHEN e1.action_owner < e1.post_owner THEN e1.action_owner ELSE e1.post_owner END AS user1_id,
        CASE WHEN e1.action_owner < e1.post_owner THEN e1.post_owner ELSE e1.action_owner END AS user2_id,
        e1.total_engagement + e2.total_engagement AS mutual_total
    FROM user_pairs e1
    JOIN user_pairs e2 ON e1.action_owner = e2.post_owner AND e1.post_owner = e2.action_owner
    WHERE e1.action_owner < e1.post_owner
    )

    SELECT 
        u1.username AS user1,
        u2.username AS user2,
        me.mutual_total AS total_mutual_engagement
    FROM mutual_engagement me
    JOIN users u1 ON me.user1_id = u1.id
    JOIN users u2 ON me.user2_id = u2.id
    ORDER BY me.mutual_total DESC
    LIMIT 3;
    """, conn)
    print(f"Connections: ")
    print(connections)
except Exception as e:
    print(f"Error: {e}")

"""
Output:
Connections: 
            user1       user2  total_mutual_engagement
0  DancingDolphin  SilverMoon                       16
1     userInBlack    TigerEye                       13
2       StarGazer  WinterWolf                       13

"""