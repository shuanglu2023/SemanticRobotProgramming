import sqlite3
from graphviz import Digraph

def extract_schema(database_path):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Extracting tables and columns
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        schema[table_name] = [column[1] for column in columns]

    # Extracting foreign key relationships
    relationships = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        fks = cursor.fetchall()
        for fk in fks:
            relationships.append((table_name, fk[2], fk[3])) # (table, foreign_table, column)

    connection.close()
    return schema, relationships

def generate_diagram(schema, relationships, output_path):
    dot = Digraph(comment='Database Schema')

    # Add nodes (tables)
    for table, columns in schema.items():
        dot.node(table, '\n'.join([table] + columns))

    # Add edges (relationships)
    for table, foreign_table, column in relationships:
        dot.edge(table, foreign_table, label=column)

    dot.render(output_path, view=True)

# Example usage
db_path ="./db/tasks.db"
schema, relationships = extract_schema(db_path)
generate_diagram(schema, relationships, 'database_schema')

