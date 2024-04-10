# db/database.py
import sqlite3

# SQL statements for creating tables
create_actions_table = """
CREATE TABLE IF NOT EXISTS Actions (
    ActionId INTEGER PRIMARY KEY,
    ActionName TEXT NOT NULL,
    TargetObjectId INTEGER,
    FOREIGN KEY (TargetObjectId) REFERENCES Products(ObjectId)
);
"""

create_products_table = """
CREATE TABLE IF NOT EXISTS Products (
    ObjectId INTEGER PRIMARY KEY,
    ObjectName TEXT NOT NULL,
    SourceLocationId INTEGER,
    TargetLocationId INTEGER,
    Color TEXT,
    TokenId INTEGER,
    SentenceId INTEGER,
    ClassId INTEGER,
    FOREIGN KEY (SourceLocationId) REFERENCES SourceLocations(LocationId),
    FOREIGN KEY (TargetLocationId) REFERENCES TargetLocations(LocationId), 
    FOREIGN KEY (ClassId) REFERENCES GroundedProducts(ClassId)
);
"""

create_grounded_products_table_string = """
CREATE TABLE IF NOT EXISTS GroundedProducts (
    ClassId INTEGER PRIMARY KEY,
    ObjectName TEXT NOT NULL,
    SourceLocationId INTEGER,
    TargetLocationId INTEGER,
    Color TEXT,
    FOREIGN KEY (SourceLocationId) REFERENCES SourceLocations(LocationId),
    FOREIGN KEY (TargetLocationId) REFERENCES TargetLocations(LocationId)
);
"""

create_source_locations_table = """
CREATE TABLE IF NOT EXISTS SourceLocations (
    LocationId INTEGER PRIMARY KEY,
    TrajectorObjectId INTEGER,
    Description TEXT,
    FOREIGN KEY (TrajectorObjectId) REFERENCES Products(ObjectId)
);
"""

create_target_locations_table = """
CREATE TABLE IF NOT EXISTS TargetLocations (
    LocationId INTEGER PRIMARY KEY,
    IndicatorObjectId INTEGER,
    Description TEXT,
    FOREIGN KEY (IndicatorObjectId) REFERENCES Products(ObjectId)
);
"""
create_target_objects_table = """
CREATE TABLE IF NOT EXISTS TaskModelTargetObjects (
    TaskModelId INTEGER,
    TargetObjectId INTEGER,
    FOREIGN KEY (TaskModelId) REFERENCES TaskModels(TaskModelId),
    FOREIGN KEY (TargetObjectId) REFERENCES Products(ObjectId)
);
"""
create_hand_positions_table = """
CREATE TABLE IF NOT EXISTS HandPositions (
    TimeId INTEGER PRIMARY KEY AUTOINCREMENT,
    TaskModelId INTEGER,
    X REAL,
    Y REAL,
    Z REAL
);
"""

def create_connection(db_path):
    """Create a database connection to the SQLite database specified by db_path"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_tables(db_path):
    """Create the tables in the database."""
    conn = create_connection(db_path)
    cursor = conn.cursor()
    # SQL create table statements here
    # cursor.execute(create_actions_table)
    cursor.execute(create_products_table)
    cursor.execute(create_source_locations_table)
    cursor.execute(create_target_locations_table)
    cursor.execute(create_target_objects_table)
    cursor.execute(create_hand_positions_table)
    conn.commit()
    conn.close()

def create_product_positions_table(db_path,ProductName):
    conn = create_connection(db_path)
    cursor = conn.cursor()
    create_time_table = f'''
    CREATE TABLE IF NOT EXISTS ProductPositions{ProductName} (
        TimeId INTEGER PRIMARY KEY AUTOINCREMENT, 
        TaskModelId INTEGER,
        ProductId INTEGER,
        SourceLocationId INTEGER,
        TargetLocationId INTEGER,
        X REAL,
        Y REAL,
        Z REAL,
        Rx REAL,
        Ry REAL,
        Rz REAL,
        Reach INTEGER,
        Grasp INTEGER,
        Move INTEGER,
        Release INTEGER,
        FOREIGN KEY (ProductId) REFERENCES Products(ObjectId),
        FOREIGN KEY (TaskModelId) REFERENCES TaskModels(TaskModelId),
        FOREIGN KEY (SourceLocationId) REFERENCES SourceLocations(LocationId),
        FOREIGN KEY (TargetLocationId) REFERENCES TargetLocations(LocationId)
    );'''
    cursor.execute(create_time_table)
    conn.commit()
    conn.close()

def create_grounded_products_table(db_path):
    conn = create_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(create_grounded_products_table_string)
    conn.commit()
    conn.close()