# Explainations for the work are being added as comments
import sqlite3
import pandas as pd

# Current db file location
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

# Read all table names -> turn it to a dataframe
tablenames_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

# Convert df to a list
tables = tablenames_df['name'].tolist()

for table in tables:
    print(f"Table: {table}")
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    # Inspect the table
    print(f"Number of rows: {len(df)}")
    print(f"Available columns: {df.columns.tolist()}")
    # Get metadata
    col = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
    for idx, row in col.iterrows():
        print(f"Name: {row['name']}")
        print(f"Type: {row['type']}")
        # Hardcoded purpose as metadata is not available in db
        print(f"Purpose: -")
        print(f"Example: {df[row['name']].head(1).values[0]}")
        print("--")
    print("-----")

"""
Output:

Table: follows
Note: This is a many-to-many relationship table between users and their followers
Number of rows: 7225
Available columns: ['follower_id', 'followed_id']
Name: follower_id
Type: INT
Purpose: This is the id of the user who is following
Example: 12
--
Name: followed_id
Type: INT
Purpose: This is the id of the user who is being followed
Example: 1
--
-----
Table: users
Number of rows: 210
Available columns: ['id', 'username', 'location', 'birthdate', 'created_at', 'profile', 'password']
Name: id
Type: INT
Purpose: Id of the user
Example: 1
--
Name: username
Type: varchar(50)
Purpose: Username of user
Example: artistic_amy
--
Name: location
Type: varchar(100)
Purpose: Location of user
Example: Boston, USA
--
Name: birthdate
Type: date
Purpose: User's date of birth
Example: 1997-06-30
--
Name: created_at
Type: timestamp
Purpose: The timestamp when the user account was created
Example: 2022-07-01 12:17:48
--
Name: profile
Type: TEXT
Purpose: Profile description of user that contains personality traits and interests
Example: Artistic soul from Boston ? | Born in '97 | Balancing mind & style | Fashion lover | News junkie | Embracing the highs and lows | Dreaming big, moving forward ✨
--
Name: password
Type: TEXT
Purpose: Password for the account
Example: izmQoLHw
--
-----
Table: sqlite_sequence
Note: Automatically created table manage AUTOINCREMENT fields
Number of rows: 3
Available columns: ['name', 'seq']
Name: name
Type:
Purpose: Shows which table (like reactions, posts, ect) the row is about
Example: reactions
--
Name: seq
Type:
Purpose: Shows the last used AUTOINCREMENT value for that table
Example: 8286
--
-----
Table: reactions
Number of rows: 8276
Available columns: ['id', 'post_id', 'user_id', 'reaction_type']
Name: id
Type: INTEGER
Purpose: Id of the reaction
Example: 1
--
Name: post_id
Type: INTEGER
Purpose: Id of the post that the reaction is for
Example: 2631
--
Name: user_id
Type: INTEGER
Purpose: Id of the user who made the reaction
Example: 60
--
Name: reaction_type
Type: TEXT
Purpose: The type of reaction
Example: like
--
-----
Table: comments
Number of rows: 5804
Available columns: ['id', 'post_id', 'user_id', 'content', 'created_at']
Name: id
Type: INTEGER
Purpose: Id of the comment
Example: 1
--
Name: post_id
Type: INTEGER
Purpose: Id of the post that the comment is for
Example: 1963
--
Name: user_id
Type: INTEGER
Purpose: Id of the user who commented
Example: 55
--
Name: content
Type: TEXT
Purpose: Content of the comment
Example: Haha, I bet your neighbors are either loving or hating you right now! Crank it up and see if you can get a dance party going next door. #DIYparty
--
Name: created_at
Type: TIMESTAMP
Purpose: The timestamp when the comment was created
Example: 2022-12-04 02:36:15
--
-----
Table: posts
Number of rows: 1303
Available columns: ['id', 'user_id', 'content', 'created_at']
Name: id
Type: INTEGER
Purpose: Id of the post
Example: 1718
--
Name: user_id
Type: INTEGER
Purpose: Id of the post owner
Example: 10
--
Name: content
Type: TEXT
Purpose: Content of the post
Example: Just had the most ridiculous encounter with a cat in Shibuya. It hissed like I was invading its turf! #CatWhisperer #TokyoLife
--
Name: created_at
Type: TIMESTAMP
Purpose: The timestamp when the post was created
Example: 2023-10-12 10:43:24
--
-----

"""

