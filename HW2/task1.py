import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
For this task, I thought about the growing factors, is it linear growth or exponential growth, as normally some social media platforms grow exponentially in the beginning, but after a while, the growth rate slows down.
After analyzing the data, I found that the growth is more linear than exponential. So I decided to use a linear projection for the next 3 years.

And the answer for the number of additional servers needed is 23. The calculation will be shown below.

"""

def get_data():
    conn = sqlite3.connect('database.sqlite')
    
    # This query get total counts of users, posts, and comments.
    totals = pd.read_sql_query("SELECT (SELECT COUNT(*) FROM users) as users, (SELECT COUNT(*) FROM posts) as posts, (SELECT COUNT(*) FROM comments) as comments", conn)

    # These queries get monthly new users, posts, and comments.
    monthly_users = pd.read_sql_query("SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count FROM users GROUP BY strftime('%Y-%m', created_at) ORDER BY month", conn)
    monthly_posts = pd.read_sql_query("SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count FROM posts GROUP BY strftime('%Y-%m', created_at) ORDER BY month", conn)
    monthly_comments = pd.read_sql_query("SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count FROM comments GROUP BY strftime('%Y-%m', created_at) ORDER BY month", conn)
    
    conn.close()
    
    return totals.iloc[0]['users'], totals.iloc[0]['posts'], totals.iloc[0]['comments'], monthly_users, monthly_posts, monthly_comments

def calculate_projections(total_users, total_posts, total_comments, monthly_users):
    # The value 1.0 is based on the assumption that each user has many props, such as posts, comments, authentication, etc.
    user_weight = 1.0

    # For posts, each of them can contains long text, images, and interactions.
    post_weight = 0.5

    # For comments, they are usually short text, but can also contain images, and reactions.
    comment_weight = 0.2

    # Traffic spike factor to account for peak times when user activity is higher.
    traffic_spike_factor = 1.2
    
    # Current server load
    current_load = (total_users * user_weight + total_posts * post_weight + total_comments * comment_weight) * traffic_spike_factor
    
    # Continue current growth for 3 years
    days_until_now = len(monthly_users) * 30
    daily_user_growth = total_users / days_until_now

    # Projected number of users for the next 3 years
    projected_users = total_users + (daily_user_growth * 1095)

    # Projected posts and comments based on user growth
    user_growth_multiplier = projected_users / total_users
    projected_posts = total_posts * user_growth_multiplier
    projected_comments = total_comments * user_growth_multiplier
    
    # Calculate projected server load and servers needed
    projected_load = (projected_users * user_weight + projected_posts * post_weight + projected_comments * comment_weight) * traffic_spike_factor

    # Current servers with 20% redundancy
    needed_servers = 16 * (projected_load / current_load) * 1.2
    
    return {
        'users': projected_users, 
        'posts': projected_posts, 
        'comments': projected_comments, 
        'needed_servers': needed_servers
    }

def create_plots(monthly_users, monthly_posts, monthly_comments):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    for df in [monthly_users, monthly_posts, monthly_comments]:
        df['date'] = pd.to_datetime(df['month'])
        df['cumulative'] = df['count'].cumsum()
    
    # Create 4 plots
    ax1.plot(monthly_users['date'], monthly_users['cumulative'], 'b-o')
    ax1.set_title('Cumulative Users'); ax1.grid(True)
    
    ax2.plot(monthly_posts['date'], monthly_posts['cumulative'], 'r-o')
    ax2.set_title('Cumulative Posts'); ax2.grid(True)
    
    ax3.plot(monthly_comments['date'], monthly_comments['cumulative'], 'g-o')
    ax3.set_title('Cumulative Comments'); ax3.grid(True)
    
    ax4.plot(monthly_users['date'], monthly_users['count'], 'b-o', label='Users/month')
    ax4.set_title('Monthly New Users'); ax4.grid(True); ax4.legend()
    
    plt.tight_layout()
    plt.savefig('growth_analysis.png', dpi=150)
    plt.show()

def analyze_and_plot():
    total_users, total_posts, total_comments, monthly_users, monthly_posts, monthly_comments = get_data()
    
    print(f"Current: {total_users} users, {total_posts} posts, {total_comments} comments")
    
    results = calculate_projections(total_users, total_posts, total_comments, monthly_users)
    
    print(f"\n3-Year Linear Projection:")
    print(f"  Users: {results['users']:.0f}, Posts: {results['posts']:.0f}, Comments: {results['comments']:.0f}")
    print(f"  Additional servers needed: +{results['needed_servers'] - 16:.0f}")
    print(f"  Total servers: {results['needed_servers']:.0f}")
    
    create_plots(monthly_users, monthly_posts, monthly_comments)
    
if __name__ == "__main__":
    analyze_and_plot()

"""
Output:
Current: 211 users, 1303 posts, 5804 comments

3-Year Linear Projection:
  Users: 431, Posts: 2662, Comments: 11857
  Additional servers needed: +23
  Total servers: 39

"""