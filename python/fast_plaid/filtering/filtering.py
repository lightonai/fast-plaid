from __future__ import annotations

import json
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


def _is_list_value(value: Any) -> bool:
    """Check if a value is a list or tuple (i.e. a multi-value field)."""
    return isinstance(value, (list, tuple))


def _detect_list_fields(metadata: list[dict[str, Any]]) -> set[str]:
    """Scan metadata and return the set of field names that contain list values.

    A field is considered a list field if ANY row has a list or tuple value for it.
    """
    list_fields: set[str] = set()
    for item in metadata:
        for key, value in item.items():
            if value is not None and _is_list_value(value):
                list_fields.add(key)
    return list_fields


def _inverted_index_path(index: str) -> str:
    """Return the file path for the inverted index JSON file."""
    return os.path.join(index, "metadata_inverted.json")


def _load_inverted_index(index: str) -> dict[str, dict[str, list[int]]]:
    """Load the inverted index from disk.

    Returns an empty dict if the file does not exist.
    """
    path = _inverted_index_path(index)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_inverted_index(index: str, inverted: dict[str, dict[str, list[int]]]) -> None:
    """Save the inverted index to disk as a JSON file."""
    path = _inverted_index_path(index)
    with open(path, "w") as f:
        json.dump(inverted, f, separators=(",", ":"))


def _build_inverted_index(
    metadata: list[dict[str, Any]],
    list_fields: set[str],
    start_id: int = 0,
) -> dict[str, dict[str, list[int]]]:
    """Build an inverted index from metadata for the given list fields.

    Args:
        metadata: The list of metadata dicts (one per document).
        list_fields: The set of field names that are list-valued.
        start_id: The _subset_ ID to assign to the first row.

    Returns:
        A dict mapping field_name -> {str(value): [subset_id, ...]}.

    """
    inverted: dict[str, dict[str, list[int]]] = {field: {} for field in list_fields}
    for i, item in enumerate(metadata):
        subset_id = start_id + i
        for field in list_fields:
            values = item.get(field)
            if values is None or not _is_list_value(values):
                continue
            for v in values:
                key = str(v)
                if key not in inverted[field]:
                    inverted[field][key] = []
                inverted[field][key].append(subset_id)
    return inverted


def _merge_inverted_indices(
    base: dict[str, dict[str, list[int]]],
    new: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, list[int]]]:
    """Merge a new inverted index into a base inverted index (mutates base).

    For each field and value, the new subset ID lists are appended to the base.
    """
    for field, mapping in new.items():
        if field not in base:
            base[field] = {}
        for value_key, subset_ids in mapping.items():
            if value_key not in base[field]:
                base[field][value_key] = []
            base[field][value_key].extend(subset_ids)
    return base


def _serialize_metadata_row(
    item: dict[str, Any],
    list_fields: set[str],
) -> dict[str, Any]:
    """Return a copy of the metadata item with list values serialized as JSON."""
    row = {}
    for key, value in item.items():
        if key in list_fields and value is not None and _is_list_value(value):
            row[key] = json.dumps(value)
        else:
            row[key] = value
    return row


def _get_list_field_names(index: str) -> set[str]:
    """Return the set of list field names from the inverted index on disk.

    Returns an empty set if no inverted index exists.
    """
    inverted = _load_inverted_index(index)
    return set(inverted.keys())


def _deserialize_list_fields_in_row(
    row: dict[str, Any],
    list_fields: set[str],
) -> dict[str, Any]:
    """Deserialize JSON-encoded list fields back to Python lists in a row dict."""
    for field in list_fields:
        if field in row and row[field] is not None and isinstance(row[field], str):
            try:
                row[field] = json.loads(row[field])
            except (json.JSONDecodeError, ValueError):
                pass
    return row


def _rebuild_inverted_index_from_db(
    index: str,
    list_fields: set[str],
) -> dict[str, dict[str, list[int]]]:
    """Rebuild the inverted index by reading list field columns from SQLite.

    This is used after delete + re-index operations where _subset_ IDs have
    changed and the inverted index must be reconstructed from scratch.
    """
    if not list_fields:
        return {}

    path = os.path.join(index, "metadata.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    columns_to_select = ", ".join(['"_subset_"'] + [f'"{f}"' for f in list_fields])
    cursor.execute(f"SELECT {columns_to_select} FROM METADATA ORDER BY _subset_")  # noqa: S608
    rows = cursor.fetchall()
    conn.close()

    inverted: dict[str, dict[str, list[int]]] = {field: {} for field in list_fields}
    for row in rows:
        subset_id = row["_subset_"]
        for field in list_fields:
            raw_value = row[field]
            if raw_value is None:
                continue
            try:
                values = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if not isinstance(values, list):
                continue
            for v in values:
                key = str(v)
                if key not in inverted[field]:
                    inverted[field][key] = []
                inverted[field][key].append(subset_id)
    return inverted


def create(
    index: str,
    metadata: list[dict[str, Any]],
) -> None:
    """Create a new SQLite database and table, erasing any existing one.

    It can handle standard Python types, including `date` and `datetime` objects.
    List and tuple values are automatically detected and stored as JSON strings
    in SQLite, with a companion inverted index file for fast lookups via
    `where_any` and `where_all`.

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

    # Also remove any existing inverted index
    inv_path = _inverted_index_path(index)
    if os.path.exists(inv_path):
        os.remove(inv_path)

    if not metadata:
        print("Warning: No metadata provided. An empty database will be created.")
        return

    # Detect list fields before anything else
    list_fields = _detect_list_fields(metadata)

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

        # List fields are always stored as TEXT (JSON-encoded strings)
        if col in list_fields:
            sql_type = "TEXT"
        else:
            # Find the first non-null value for the column to infer its type
            value = next(
                (
                    item[col]
                    for item in metadata
                    if col in item and item[col] is not None
                ),
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
        serialized = _serialize_metadata_row(item, list_fields)
        row = [i] + [serialized.get(col) for col in columns]
        rows_to_insert.append(tuple(row))

    cursor.executemany(insert_sql, rows_to_insert)

    conn.commit()
    conn.close()

    # Build and save the inverted index for list fields
    if list_fields:
        inverted = _build_inverted_index(metadata, list_fields, start_id=0)
        _save_inverted_index(index, inverted)

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

    List fields in the new metadata are auto-detected and merged into the
    existing inverted index.

    Args:
    ----
        index: The path to the index directory containing the database.
        metadata: A list of dictionaries representing the new rows to insert.

    """
    if not metadata:
        print("No metadata provided to update.")
        return

    # Detect list fields in the new metadata and merge with existing ones
    new_list_fields = _detect_list_fields(metadata)
    existing_inverted = _load_inverted_index(index)
    existing_list_fields = set(existing_inverted.keys())
    all_list_fields = existing_list_fields | new_list_fields

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

        # List fields are always TEXT
        if column in all_list_fields:
            sql_type = "TEXT"
        else:
            first_value = next(
                (item[column] for item in metadata if column in item), None
            )
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
        serialized = _serialize_metadata_row(item, all_list_fields)
        row = [new_id] + [serialized.get(col) for col in db_columns]
        rows_to_insert.append(tuple(row))

    cursor.executemany(insert_sql, rows_to_insert)
    conn.commit()
    conn.close()

    # Update the inverted index with the new rows
    if all_list_fields:
        new_inverted = _build_inverted_index(
            metadata, all_list_fields, start_id=start_id
        )
        merged = _merge_inverted_indices(existing_inverted, new_inverted)
        _save_inverted_index(index, merged)


def delete(index: str, subset: list[int] | int) -> None:
    """Delete rows from the database and re-index the `_subset_` column.

    After deleting the specified rows, this function recomputes the entire
    `_subset_` column for the remaining rows, ensuring it is sequential
    and starts from 0. The inverted index (if any) is rebuilt from scratch
    to reflect the new IDs.

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

    # Load inverted index to know which fields are list fields
    list_fields = _get_list_field_names(index)

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

    # Rebuild the inverted index from the re-indexed SQLite table.
    # This is necessary because all _subset_ IDs have been reassigned.
    if list_fields:
        inverted = _rebuild_inverted_index_from_db(index, list_fields)
        _save_inverted_index(index, inverted)


def get(
    index: str,
    condition: str | None = None,
    parameters: tuple | list = (),
    subset: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve rows as a list of dictionaries, filtered by condition or subset.

    This function allows filtering by either a SQL `condition` or a list of
    `_subset_` IDs. List fields that were stored as JSON strings are
    automatically deserialized back to Python lists.

    **Ordering behavior:**
    - If `subset` is provided: Returns rows ordered exactly as they appear in the
      `subset` list (including duplicates).
    - If `condition` is provided or `subset` is None: Returns rows ordered by
      `_subset_` (ascending ID).

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

    should_sort_by_subset = False

    if condition is not None:
        query += f" WHERE {condition}"
        query += " ORDER BY _subset_"
    elif subset is not None:
        if not subset:
            return []
        placeholders = ", ".join(["?"] * len(subset))
        query += f" WHERE _subset_ IN ({placeholders})"
        params = subset
        should_sort_by_subset = True
    else:
        query += " ORDER BY _subset_"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Convert sqlite3.Row objects to standard Python dictionaries
    results = [dict(row) for row in rows]

    # If subset was provided, re-order (and potentially duplicate) results
    # to match the input subset list
    if should_sort_by_subset and subset is not None:
        results_map = {row["_subset_"]: row for row in results}
        # Retrieve items in the order of 'subset', skipping any that weren't found
        results = [results_map[i] for i in subset if i in results_map]

    # Deserialize list fields back to Python lists
    list_fields = _get_list_field_names(index)
    if list_fields:
        results = [_deserialize_list_fields_in_row(row, list_fields) for row in results]

    return results


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


def where_any(
    index: str,
    field: str,
    values: list[Any],
) -> list[int]:
    """Retrieve _subset_ IDs where the list field contains ANY of the given values.

    This uses the pre-built inverted index for O(1)-per-value lookups.
    The result is the union of all document sets (OR semantics).

    Args:
    ----
        index: The path to the index directory.
        field: The name of the list field to query (e.g. ``"foreign_ids"``).
        values: A list of values to search for. A document is included if its
            list contains **at least one** of these values.

    Returns:
    -------
        A sorted list of `_subset_` IDs matching the condition.

    """
    inv_path = _inverted_index_path(index)
    if not os.path.exists(inv_path):
        error = (
            "No inverted index found. This index has no list fields, "
            "or was created before inverted index support was added."
        )
        raise FileNotFoundError(error)

    inverted = _load_inverted_index(index)

    if field not in inverted:
        available = list(inverted.keys()) if inverted else []
        error = (
            f"Field '{field}' is not a known list field. "
            f"Available list fields: {available}"
        )
        raise KeyError(error)

    field_index = inverted[field]

    result_set: set[int] = set()
    for v in values:
        key = str(v)
        if key in field_index:
            result_set.update(field_index[key])

    return sorted(result_set)


def where_all(
    index: str,
    field: str,
    values: list[Any],
) -> list[int]:
    """Retrieve _subset_ IDs where the list field contains ALL of the given values.

    This uses the pre-built inverted index for O(1)-per-value lookups.
    The result is the intersection of all document sets (AND semantics).

    Args:
    ----
        index: The path to the index directory.
        field: The name of the list field to query (e.g. ``"foreign_ids"``).
        values: A list of values to search for. A document is included only if
            its list contains **every one** of these values.

    Returns:
    -------
        A sorted list of `_subset_` IDs matching the condition.

    """
    inv_path = _inverted_index_path(index)
    if not os.path.exists(inv_path):
        error = (
            "No inverted index found. This index has no list fields, "
            "or was created before inverted index support was added."
        )
        raise FileNotFoundError(error)

    inverted = _load_inverted_index(index)

    if field not in inverted:
        available = list(inverted.keys()) if inverted else []
        error = (
            f"Field '{field}' is not a known list field. "
            f"Available list fields: {available}"
        )
        raise KeyError(error)

    if not values:
        return []

    field_index = inverted[field]

    result_set: set[int] | None = None
    for v in values:
        key = str(v)
        if key in field_index:
            ids = set(field_index[key])
        else:
            # If any value has no matches, intersection is empty
            return []
        if result_set is None:
            result_set = ids
        else:
            result_set &= ids
        if not result_set:
            return []

    return sorted(result_set) if result_set else []
