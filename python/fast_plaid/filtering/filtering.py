from __future__ import annotations

import os
import re
import sqlite3
from datetime import date, datetime
from typing import Any


def _prevent_sql_injection(name: str) -> bool:
    """Prevents SQL injection by ensuring the column name is a valid identifier."""
    return re.match(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$", string=name) is not None


def _get_sql_type(value: Any) -> str:
    """Infers the SQL data type from a Python value."""
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    if isinstance(value, (datetime, date)):
        return "TEXT"
    if isinstance(value, str):
        return "TEXT"
    return "BLOB"


def create(
    index: str,
    metadata: list[dict[str, Any]],
) -> None:
    """Create a new SQLite database and table, erasing any existing one.

    It can handle standard Python types, including `date` and `datetime` objects.

    Args:
    ----
        index: The path to the index directory where the database will be created.
        metadata: A list of dictionaries, where each dictionary represents
            a row to be inserted.

    Examples:
    --------
    >>> from fast_plaid import filtering
    >>> import shutil
    >>> from datetime import date

    >>> metadata = [
    ...     {"name": "Alice", "age": 30, "height": 5.5, "join_date": date(2020, 5, 17)},
    ...     {"name": "Bob", "age": 25, "height": 6.0, "join_date": date(2021, 6, 21), "town": "New York"},
    ... ]

    >>> filtering.create(
    ...     index="test_index",
    ...     metadata=metadata,
    ... )
    Database created at 'test_index/metadata.db' with 2 rows.

    >>> filtering.get(
    ...     index="test_index",
    ... )
    [{'_subset_': 0, 'name': 'Alice', 'age': 30, 'height': 5.5, 'join_date': '2020-05-17', 'town': None}, {'_subset_': 1, 'name': 'Bob', 'age': 25, 'height': 6.0, 'join_date': '2021-06-21', 'town': 'New York'}]

    >>> filtering.where(
    ...     index="test_index",
    ...     condition="age < ? AND height >= ? AND join_date > ?",
    ...     parameters=(28, 6.0, '2021-01-01'),
    ... )
    [1]

    >>> filtering.delete(
    ...     index="test_index",
    ...     subset=[1],
    ... )
    1 rows deleted from 'test_index/metadata.db'. The table has been re-indexed.

    >>> filtering.where(
    ...     index="test_index",
    ...     condition="age < ? AND height >= ? AND join_date > ?",
    ...     parameters=(28, 6.0, '2021-01-01'),
    ... )
    []

    >>> filtering.where(
    ...     index="test_index",
    ...     condition="age > ?",
    ...     parameters=[28],
    ... )
    [0]

    >>> new_metadata = [
    ...     {"name": "Charlie", "age": 35, "height": 5.8,
    ...     "join_date": date(2019, 3, 15), "first_name": "Charles"}
    ... ]

    >>> filtering.update(
    ...     index="test_index",
    ...     metadata=new_metadata,
    ... )
    Added new column 'first_name' with type TEXT to the table.
    1 new rows successfully added to 'test_index/metadata.db'.

    >>> filtering.where(
    ...     index="test_index",
    ...     condition="age > ?",
    ...     parameters=[28],
    ... )
    [0, 1]

    >>> filtering.get(
    ...     index="test_index",
    ...     condition="age > ?",
    ...     parameters=[28],
    ... )
    [{'_subset_': 0, 'name': 'Alice', 'age': 30, 'height': 5.5, 'join_date': '2020-05-17', 'town': None, 'first_name': None}, {'_subset_': 1, 'name': 'Charlie', 'age': 35, 'height': 5.8, 'join_date': '2019-03-15', 'town': None, 'first_name': 'Charles'}]

    >>> filtering.delete(
    ...     index="test_index",
    ...     subset=[0],
    ... )
    1 rows deleted from 'test_index/metadata.db'. The table has been re-indexed.

    >>> filtering.where(
    ...     index="test_index",
    ...     condition="age > ?",
    ...     parameters=[28],
    ... )
    [0]

    >>> filtering.get(
    ...     index="test_index",
    ...     condition="age > ?",
    ...     parameters=[28],
    ... )
    [{'_subset_': 0, 'name': 'Charlie', 'age': 35, 'height': 5.8, 'join_date': '2019-03-15', 'town': None, 'first_name': 'Charles'}]

    >>> shutil.rmtree("test_index")

    """  # noqa: E501
    if not os.path.isdir(index):
        os.makedirs(index, exist_ok=True)

    path = os.path.join(index, "metadata.db")
    if os.path.exists(path):
        os.remove(path)

    if not metadata:
        print("Warning: No metadata provided. An empty database will be created.")
        return

    # Collect all unique column names from the metadata, preserving insertion order
    all_keys: dict[str, None] = {}
    for item in metadata:
        for key in item:
            all_keys[key] = None
    columns = list(all_keys)

    col_defs = []
    for col in columns:
        if not _prevent_sql_injection(col):
            error = f"""
            Invalid column name '{col}'. Column names must start with a letter or
            underscore, followed by letters, digits, or underscores, and
            cannot contain spaces or special characters.
            """.strip()
            raise ValueError(error)

        # Find the first non-null value for the column to infer its type
        value = next(
            (item[col] for item in metadata if col in item and item[col] is not None),
            None,
        )
        sql_type = _get_sql_type(value)
        col_defs.append(f'"{col}" {sql_type}')

    # Add the special _subset_ column as the primary key
    col_defs.insert(0, '"_subset_" INTEGER PRIMARY KEY')

    # The sqlite3 module can automatically handle date/datetime objects
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    create_table_sql = f"CREATE TABLE METADATA ({', '.join(col_defs)})"
    cursor.execute(create_table_sql)

    placeholders = ", ".join(["?"] * (len(columns) + 1))
    column_names = ", ".join(f'"{c}"' for c in columns)
    insert_sql = (
        f"INSERT INTO METADATA (_subset_, {column_names}) VALUES ({placeholders})"  # noqa: S608
    )

    rows_to_insert = []
    for i, item in enumerate(metadata):
        # The first value is the _subset_ id
        row = [i] + [item.get(col) for col in columns]
        rows_to_insert.append(tuple(row))

    cursor.executemany(insert_sql, rows_to_insert)

    conn.commit()
    conn.close()
    message = f"""
    Database created at '{path}' with {len(metadata)} rows.
    """.strip()
    print(message)


def update(
    index: str,
    metadata: list[dict[str, Any]],
) -> None:
    """Append new rows to the database, adding new columns if necessary.

    This function checks for any new columns in the provided metadata that are
    not in the existing table. If new columns are found, it alters the table
    to add them. It then appends the new metadata records, automatically
    assigning incremented '_subset_' IDs.

    Args:
    ----
        index: The path to the index directory containing the database.
        metadata: A list of dictionaries representing the new rows to insert.

    """
    if not metadata:
        print("No metadata provided to update.")
        return

    path = os.path.join(index, "metadata.db")
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(METADATA)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns_all: set[str] = set()
    for item in metadata:
        new_columns_all.update(item.keys())

    columns_to_add = new_columns_all - existing_columns
    for column in columns_to_add:
        if not _prevent_sql_injection(column):
            error = f"""
            Invalid column name '{column}'. Column names must start with a letter or
            underscore, followed by letters, digits, or underscores, and
            cannot contain spaces or special characters.
            """.strip()
            raise ValueError(error)

        first_value = next((item[column] for item in metadata if column in item), None)
        sql_type = _get_sql_type(first_value)

        alter_sql = f'ALTER TABLE METADATA ADD COLUMN "{column}" {sql_type}'
        cursor.execute(alter_sql)
        print(f"Added new column '{column}' with type {sql_type} to the table.")

    cursor.execute("SELECT MAX(_subset_) FROM METADATA")
    max_id = cursor.fetchone()[0]
    start_id = (max_id + 1) if max_id is not None else 0

    # Re-fetch column order after potential alterations
    cursor.execute("PRAGMA table_info(METADATA)")
    db_columns = [row[1] for row in cursor.fetchall() if row[1] != "_subset_"]

    placeholders = ", ".join(["?"] * (len(db_columns) + 1))

    # The S608 warning is a false positive as column names are from a trusted source.
    column_names = ", ".join(f'"{c}"' for c in db_columns)
    insert_sql = (
        f"INSERT INTO METADATA (_subset_, {column_names}) VALUES ({placeholders})"  # noqa: S608
    )

    rows_to_insert = []
    for i, item in enumerate(metadata):
        new_id = start_id + i
        # Use item.get(col) to handle missing keys gracefully (inserts NULL)
        row = [new_id] + [item.get(col) for col in db_columns]
        rows_to_insert.append(tuple(row))

    cursor.executemany(insert_sql, rows_to_insert)
    conn.commit()
    conn.close()

    print(f"{len(metadata)} new rows successfully added to '{path}'.")


def delete(index: str, subset: list[int] | int) -> None:
    """Delete rows from the database and re-index the `_subset_` column.

    After deleting the specified rows, this function recomputes the entire
    `_subset_` column for the remaining rows, ensuring it is sequential
    and starts from 0.

    Args:
    ----
        index: The path to the index directory containing the database.
        subset: A list of `_subset_` IDs to delete. The list must be sorted
            in ascending order.

    """
    if isinstance(subset, int):
        subset = [subset]

    # Assert that all elements are integers.
    assert all(isinstance(i, int) for i in subset), (
        "All elements in the 'subset' list must be integers."
    )
    # Assert that the list of IDs to delete is sorted.
    assert all(subset[i] <= subset[i + 1] for i in range(len(subset) - 1)), (
        "The 'subset' list of IDs to delete must be sorted in ascending order."
    )

    if not subset:
        print("0 rows deleted (empty subset provided).")
        return

    path = os.path.join(index, "metadata.db")
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    try:
        # Use a transaction to ensure atomicity for the delete and re-index operations.
        cursor.execute("BEGIN")

        placeholders = ", ".join(["?"] * len(subset))
        # The S608 warning is a false positive as placeholders are generated safely.
        delete_sql = f"DELETE FROM METADATA WHERE _subset_ IN ({placeholders})"  # noqa: S608
        cursor.execute(delete_sql, subset)
        rows_affected = cursor.rowcount

        cursor.execute("PRAGMA table_info(METADATA)")
        columns = [row[1] for row in cursor.fetchall() if row[1] != "_subset_"]
        column_str = ", ".join(f'"{c}"' for c in columns)

        # Use the ROW_NUMBER() window function to generate new, sequential IDs
        # starting from 0. The existing `_subset_` order is preserved.
        create_temp_sql = f"""
            CREATE TEMP TABLE METADATA_TEMP AS
            SELECT
                (ROW_NUMBER() OVER (ORDER BY _subset_)) - 1 AS new_subset_id,
                {column_str}
            FROM METADATA
        """  # noqa: S608
        cursor.execute(create_temp_sql)

        cursor.execute("DELETE FROM METADATA")

        insert_back_sql = f"""
            INSERT INTO METADATA (_subset_, {column_str})
            SELECT new_subset_id, {column_str} FROM METADATA_TEMP
        """  # noqa: S608
        cursor.execute(insert_back_sql)
        cursor.execute("DROP TABLE METADATA_TEMP")
        conn.commit()

        message = f"{rows_affected} rows deleted from '{path}'. The table has been re-indexed."  # noqa: E501
        print(message)

    except Exception as e:
        # If any error occurs, roll back all changes to maintain data integrity.
        conn.rollback()
        print(f"An error occurred: {e}. The transaction was rolled back.")
        raise e
    finally:
        # Always close the connection.
        conn.close()


def get(
    index: str,
    condition: str | None = None,
    parameters: tuple | list = (),
    subset: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve rows as a list of dictionaries, filtered by condition or subset.

    This function allows filtering by either a SQL `condition` or a list of
    `_subset_` IDs, but not both. If neither is provided, all rows are returned.

    Args:
    ----
        index: The path to the index directory.
        condition: An optional SQL WHERE clause with '?' placeholders.
        parameters: A tuple or list of values for the '?' placeholders in the condition.
        subset: An optional list of `_subset_` IDs to retrieve.

    Returns:
    -------
        A list of dictionaries, where each dictionary represents a row.

    Raises:
    ------
        ValueError: If both `condition` and `subset` are provided.

    """
    if condition is not None and subset is not None:
        error = "Please provide either a 'condition' or a 'subset', not both."
        raise ValueError(error)

    path = os.path.join(index, "metadata.db")
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    cursor = conn.cursor()

    query = "SELECT * FROM METADATA"
    params = parameters

    if condition is not None:
        query += f" WHERE {condition}"
    elif subset is not None:
        if not subset:
            return []
        placeholders = ", ".join(["?"] * len(subset))
        query += f" WHERE _subset_ IN ({placeholders})"
        params = subset

    query += " ORDER BY _subset_"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Convert sqlite3.Row objects to standard Python dictionaries
    return [dict(row) for row in rows]


def where(
    index: str,
    condition: str,
    parameters: tuple | list = (),
) -> list[int]:
    """Retrieve a list of _subset_ IDs that match a given SQL condition.

    Args:
    ----
        index: The path to the index directory.
        condition: A SQL WHERE clause with '?' placeholders for values.
        parameters: A tuple or list of values to substitute for the '?'
            placeholders. Defaults to an empty tuple.

    Returns:
    -------
        A list of `_subset_` IDs that match the condition.

    """
    path = os.path.join(index, "metadata.db")

    if not os.path.exists(path):
        error = """No metadata database found. Please create it first by
        adding metadata during index creation.
        """.strip()
        raise FileNotFoundError(error)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    query = f"SELECT _subset_ FROM METADATA WHERE {condition}"  # noqa: S608
    cursor.execute(query, parameters)

    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results
