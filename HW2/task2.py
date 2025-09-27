# Explainations for the work are being added as comments
import sqlite3
import pandas as pd

# Current db file location
dbfile = 'database.sqlite'
# Establish a connection to the db
conn = sqlite3.connect(dbfile)

"""
After research about viral post, I found that it is a piece of content that gets shared quickly across various social media platforms in a short period of time. So I decided to use growth rate in the first few hours to measure the virality of a post.

I was trying to calculate the growth rate based on reactions, but I found that the table does not have a created_at column, so I can only use comments in this case.
"""

CALCULATING_HOURS = 24

def calculate_growth_rate_hours(table_alias, post_alias, hours):
    # Don't forget to check if the hours since posted is less than the calculating hours
    return f"""
    COUNT(DISTINCT CASE WHEN (julianday({table_alias}.created_at) - julianday({post_alias}.created_at)) * 24 <= {hours} THEN {table_alias}.id END) * 1.0 / 
    CASE 
        WHEN (julianday('now') - julianday({post_alias}.created_at)) * 24 >= {hours} THEN {hours}
        WHEN (julianday('now') - julianday({post_alias}.created_at)) * 24 < 1 THEN 1
        ELSE (julianday('now') - julianday({post_alias}.created_at)) * 24
    END
    """

try:
    viral_post_df = pd.read_sql_query(f"""
    SELECT
        p.id,

        -- Total engagement (comments + reactions)
        COUNT(DISTINCT c.id) as total_comments,
        COUNT(DISTINCT r.id) as total_reactions,
        (COUNT(DISTINCT c.id) + COUNT(DISTINCT r.id)) as absolute_engagement,

        -- Growth rate: comments per hour in first {CALCULATING_HOURS} hours
        {calculate_growth_rate_hours('c', 'p', CALCULATING_HOURS)} as growth_rate,
        
        -- Combined virality score
        {calculate_growth_rate_hours('c', 'p', CALCULATING_HOURS)} * (COUNT(DISTINCT c.id) + COUNT(DISTINCT r.id)) as virality_score
    FROM posts p
    LEFT JOIN comments c on c.post_id = p.id
    LEFT JOIN reactions r on r.post_id = p.id
    GROUP by p.id
    HAVING absolute_engagement > 0
    ORDER BY virality_score DESC
    LIMIT 3;
    """, conn)
    print(f"Viral posts - first {CALCULATING_HOURS} hours: ")
    print(viral_post_df)
except Exception as e:
    print(f"Error: {e}")

"""
Output:

Viral posts - first 5 hours: 
     id  total_comments  total_reactions  absolute_engagement  growth_rate  virality_score
0  2351              62              139                  201         12.4          2492.4
1  2813              82              103                  185         12.0          2220.0
2  2195              45              133                  178          9.0          1602.0

Viral posts - first 12 hours: 
     id  total_comments  total_reactions  absolute_engagement  growth_rate  virality_score
0  2813              82              103                  185     6.833333     1264.166667
1  2351              62              139                  201     5.166667     1038.500000
2  2004              71               94                  165     5.916667      976.250000

Viral posts - first 24 hours: 
     id  total_comments  total_reactions  absolute_engagement  growth_rate  virality_score
0  2813              82              103                  185     3.416667      632.083333
1  2351              62              139                  201     2.583333      519.250000
2  2004              71               94                  165     2.958333      488.125000

As we can see, the vital posts are consistent across different hours, so the answer for the question is post id 2813, 2351, and 2004. There was a slight change in the order because there's a higher early burst of post id 2351 at the start, but slower sustained growth.
"""