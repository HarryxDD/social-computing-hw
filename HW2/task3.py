# Explainations for the work are being added as comments
import sqlite3
import pandas as pd

# Current db file location
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

"""
For this task, I define the engagement based on comments sice the reactions table does not have a created_at column.

I excluded posts that have no comments, since they do not have any engagement, but still show the number of such posts in the output.

Basically, I created a CTE to calculate the time to first comment and time to last comment for each post, then I used aggregate functions to get the required metrics.

I used INNER JOIN to exclude posts with no comments first, then I calculated the number of such posts by subtracting from the total.
"""

try:
    content_lifecycle = pd.read_sql_query(f"""
    with post_lifecycle as (
    SELECT
  	    p.id,
  	    p.created_at,
  	    MIN(c.created_at) AS first_comment_at,
  	    (julianday(MIN(c.created_at)) - julianday(p.created_at)) * 24 as hours_to_first_comment,
  	    MAX(c.created_at) as last_comment_at,
  	    (julianday(MAX(c.created_at)) - julianday(p.created_at)) * 24 as hours_to_last_comment
    from posts p
    INNER join comments c on p.id = c.post_id
    GROUP by p.id
    )
    SELECT
	    COUNT(*) as posts_with_comments,
        (select COUNT(*) from posts) - count(*) as posts_with_no_comments,
        AVG(hours_to_first_comment) as avg_hr_to_first_cmt,
        AVG(hours_to_last_comment) as avg_hr_to_last_cmt
    from post_lifecycle;
    """, conn)
    print(f"Content Lifecycle: ")
    print(content_lifecycle)
except Exception as e:
    print(f"Error: {e}")

"""
Output:
Content Lifecycle: 
   posts_with_comments  posts_with_no_comments  avg_hr_to_first_cmt  avg_hr_to_last_cmt
0                 1215                      88            86.604362          151.445664

"""