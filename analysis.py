import pandas as pd
import matplotlib.pyplot as plt
import textwrap  # Import textwrap module

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')


sql_query = "select building_id,  category from nutz_building"

df = pd.read_sql(sql_query, engine)

df['category'] = df['category'].str.replace('Infrastructure', '')
category_counts = df['category'].value_counts()
print(category_counts)


# # Creating a larger plot
# plt.figure(figsize=(10, 6))  # Adjust the figure size
#
# # Plotting
# category_counts.plot(kind='bar')
# plt.xlabel('nutz division')
# plt.ylabel('Number of Buildings')
# plt.title('Number of Buildings by nutz division')
# plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate, adjust alignment, and font size
# plt.tight_layout()  # Adjust layout to make room for label
# plt.show()

# category_counts.plot(kind='barh')  # 'barh' for horizontal bars
# plt.ylabel('Category')
# plt.xlabel('Number of Buildings')
# plt.title('Number of Buildings by Category')
# plt.show()


# Function to wrap text
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_yticklabels(labels)

# Plotting
plt.figure(figsize=(10, 8))  # Adjust the figure size
category_counts.plot(kind='barh')
plt.xlabel('Number of Buildings')
plt.title('Number of Buildings by Category')
wrap_labels(plt.gca(), 20)  # Wrap text with 20 characters width
plt.tight_layout()  # Adjust layout
plt.show()