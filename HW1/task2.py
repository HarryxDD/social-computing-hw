# Explainations for the work are being added as comments
import sqlite3
import pandas as pd

# Current db file location
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

try:
    # Check for users who not exist in posts, comments, and reactions table using subqueries
    lurkers = pd.read_sql_query("""
    SELECT
        id
    FROM users
    WHERE id NOT IN (SELECT user_id FROM posts)
    AND id NOT IN (SELECT user_id FROM comments)
    AND id NOT IN (SELECT user_id FROM reactions);
    """, conn)
    # print("Lurkers: ")
    # print(lurkers)
    print("The number of people who have not interacted at all: ", len(lurkers))
except Exception as e:
    print(f"Error: {e}")

"""
Output:
The number of people who have not interacted at all:  55

"""