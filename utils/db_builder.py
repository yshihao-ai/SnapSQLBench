import sqlite3
import logging
import os
import re
import json

from typing import Dict, List, Any, Tuple, Optional
from .constants import PipelineStatus
logger = logging.getLogger('YSHLogger')

class DatabaseBuilder:
    def __init__(self, output_dir: str): 
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_name(self, name: str, is_table=False) -> str:
        # Basic sanitization, might need improvement based on real data
        name = re.sub(r'[^\w]', '_', str(name)) # Replace non-alphanumeric with underscore
        if not name: name = "unnamed"
        if name[0].isdigit(): name = '_' + name
        keywords = {'select', 'from', 'where', 'table', 'index', 'on', 'insert', 'delete', 'update', 'create', 'drop', 'text', 'integer', 'real'}
        if name.lower() in keywords: name += "_tbl" if is_table else "_col"
        return name

    def build_db_from_schema_and_data(
        self, 
        schema_ddl: str, 
        tables_data: Dict[str, Dict[str, List[Any]]], 
        db_id: str, 
        entry_index: int,
        suffix: str = "pred" 
    ) -> Optional[str]:
        if not schema_ddl or not tables_data:
            logger.warning(f"Entry {entry_index} ({suffix}): Missing Schema DDL or table data dictionary, cannot build database.")
            return None
        
        db_filename = f"{db_id}_{entry_index}_{suffix}.sqlite"
        db_path = os.path.join(self.output_dir, db_filename)

        if os.path.exists(db_path):
            try: os.remove(db_path)
            except OSError as e: logger.warning(f"Unable to delete the old temporary database {db_path}: {e}")

        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            logger.info(f"Entry {entry_index}: Executing Schema DDL (suffix: {suffix}):n{schema_ddl}")
            cursor.executescript(schema_ddl) 

            inserted_rows_count_total = 0
            for table_name, table_content in tables_data.items():
                if not isinstance(table_content, dict):
                    logger.warning(f"Entry {entry_index} ({suffix}): The data format of table '{table_name}' is invalid (not a dictionary), skipping.")
                    continue
                
                headers = table_content.get('headers', [])
                rows = table_content.get('rows', [])
                
                if not headers:
                    logger.warning(f"Entry {entry_index} ({suffix}): Table '{table_name}' is missing 'headers', cannot insert data.")
                    continue
                if not rows:
                    logger.info(f"Entry {entry_index} ({suffix}): Table '{table_name}' has no 'rows' data, skipping insertion.")
                    continue

                try:
                    placeholders = ", ".join(["?"] * len(headers))
                    quoted_headers = ", ".join([f'"{h}"' for h in headers])
                    insert_sql = f'INSERT INTO "{table_name}" ({quoted_headers}) VALUES ({placeholders});'
                    processed_rows = []
                    for row in rows:
                        if len(row) == len(headers):
                            processed_rows.append(tuple(map(str, row)))
                        else:
                            logger.warning(f"Entry {entry_index} ({suffix}) (Table {table_name}): Skipping row that does not match header length. Expected {len(headers)}, got {len(row)}.")

                    if processed_rows:
                        cursor.executemany(insert_sql, processed_rows)
                        logger.info(f"Entry {entry_index} ({suffix}): Successfully inserted {len(processed_rows)} rows into table '{table_name}'.")
                        inserted_rows_count_total += len(processed_rows)
                
                except sqlite3.Error as e:
                    logger.error(f"Entry {entry_index} ({suffix}): Failed to insert data into table '{table_name}': {e}. SQL (template): {insert_sql}")

            conn.commit()
            logger.info(f"Entry {entry_index}: Successfully built temporary database (suffix: {suffix}), a total of {inserted_rows_count_total} rows inserted: {db_path}")
            return db_path

        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Entry {entry_index} ({suffix}): Error occurred while building database {db_path} with Schema DDL and data: {e}", exc_info=True)
            if conn: conn.rollback()
            if os.path.exists(db_path): 
                try: os.remove(db_path) 
                except OSError: pass
            return None
        finally:
            if conn: conn.close()


    def build_db_from_schema_and_inserts(
        self,
        schema_ddl: str,
        insert_statements_dict: Dict[str, List[str]],
        db_id: str,
        entry_index: int,
        suffix: str = "answer_key"
    ) -> Optional[str]:
        """
        (This method is used to construct DB_Answer_Key)
        Build the database based on the GT Schema DDL and the list of GT INSERT statements.
        - schema_ddl: a string containing one or more CREATE TABLE statements.
        - insert_statements_dict: { "table_name_1": ["INSERT...", "INSERT..."], ... }
        """
        if not schema_ddl or not insert_statements_dict:
            logger.warning(f"Entry {entry_index} ({suffix}): Missing Schema DDL or INSERT statement dictionary, unable to build the database.")
            return None

        db_filename = f"{db_id}_{entry_index}_{suffix}.sqlite"
        db_path = os.path.join(self.output_dir, db_filename)

        if os.path.exists(db_path):
            try: os.remove(db_path)
            except OSError as e: logger.warning(f"Unable to delete the old temporary database {db_path}: {e}")

        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            logger.info(f"Entry {entry_index}: Executing GT Schema DDL (suffix: {suffix}):\n{schema_ddl}")
            cursor.executescript(schema_ddl)

            inserted_rows_count_total = 0
            for table_name, inserts in insert_statements_dict.items():
                if not inserts:
                    logger.info(f"Entry {entry_index} ({suffix}): Table '{table_name}' has no INSERT statements.")
                    continue
                
                logger.info(f"Entry {entry_index} ({suffix}): Inserting {len(inserts)} rows into table '{table_name}' (from visual_data.json)...")
                try:
                    for insert_sql in inserts:
                        cursor.execute(insert_sql)
                    inserted_rows_count_total += len(inserts)
                except sqlite3.Error as e:
                    logger.error(f"Entry {entry_index} ({suffix}): Failed to insert GT data into table '{table_name}': {e}. SQL (example): {inserts[0]}")

            conn.commit()
            logger.info(f"Entry {entry_index}: Successfully built temporary database (suffix: {suffix}), a total of {inserted_rows_count_total} rows inserted: {db_path}")
            return db_path

        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Entry {entry_index} ({suffix}): Error occurred while building database {db_path} using Schema DDL and INSERT list: {e}", exc_info=True)
            if conn: conn.rollback()
            if os.path.exists(db_path): 
                try: os.remove(db_path) 
                except OSError: pass
            return None
        finally:
            if conn: conn.close()

    def fetch_db_content(self, db_path, table_names):
        content = {}
        if not db_path or not os.path.exists(db_path): 
            return content
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]
            for t in tables:
                cursor.execute(f'SELECT * FROM "{t}"')
                if not table_names or t in table_names:
                    content[t] = cursor.fetchall()
        except Exception as e:
            logger.error(f"fetch_db_content error: {e}", exc_info=True)
            raise e
        finally: conn.close()
        return content
    
    def build_db_from_data(
        self, 
        db_id: str, 
        entry_index: int,
        table_name: str,
        table_data: Optional[Dict[str, Any]], 
    ):
        """
        Logic:
        1. `schema_sql` must exist.
        2. If DDL execution fails, directly report an error and skip.
        """
        if not table_data:
            return None, None, PipelineStatus.P1_MODEL_OUTPUT_EMPTY
            
        if isinstance(table_data, str):
            logger.warning(f"Item {db_id} table '{table_name}' data is string, trying to parse...")
            try:
                clean_str = table_data.replace("```json", "").replace("```", "").strip()
                table_data = json.loads(clean_str)
            except Exception as e:
                logger.error(f"Failed to parse table data string for Item {db_id}: {e}")
                return None, None, PipelineStatus.P1_MODEL_OUTPUT_INVALID

        if not isinstance(table_data, dict):
            logger.error(f"Item {db_id} table '{table_name}' is not a dict (Got {type(table_data)}). Skipping.")
            return None, None, PipelineStatus.P1_MODEL_OUTPUT_INVALID
        
        
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        predicted_ddl = table_data.get('schema_sql', "")

        target_table_name = table_name

        db_subdir = os.path.join(self.output_dir, str(db_id))
        os.makedirs(db_subdir, exist_ok=True)
        db_filename = f"{db_id}.sqlite"
        db_path = os.path.join(db_subdir, db_filename)
        conn = None

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(f'DROP TABLE IF EXISTS "{target_table_name}"')
            except Exception:
                None, None, PipelineStatus.P1_UNKNOWN_ERROR

            if not predicted_ddl:
                logger.error(f"Entry {entry_index}: [Strict Mode] Table '{target_table_name}' is missing schema_sql, skipping table creation.")
                return None, None, PipelineStatus.P1_SCHEMA_CREATE_FAILED

            try:
                cursor.execute(predicted_ddl)
                logger.info(f"Entry {entry_index}: [Schema Success] Executing model DDL ({target_table_name})")
            except sqlite3.Error as e:
                logger.error(f"Entry {entry_index}: [Strict Mode] Model DDL execution failed: {e}. SQL: {predicted_ddl}")
                return None, None, PipelineStatus.P1_SCHEMA_CREATE_FAILED

            if rows:
                cursor.execute(f"PRAGMA table_info(\"{target_table_name}\")")
                table_info = cursor.fetchall() # List of (cid, name, type, notnull, dflt_value, pk)
                num_columns = len(table_info)
                
                if num_columns > 0:
                    placeholders = ", ".join(["?"] * num_columns)
                    insert_sql = f'INSERT INTO "{target_table_name}" VALUES ({placeholders})'
                    logger.info(insert_sql)
                    processed_rows = []
                    for r in rows:
                        current_row = r
                        if len(r) > num_columns:
                            current_row = r[:num_columns]
                        elif len(r) < num_columns:
                            current_row = r + [None] * (num_columns - len(r))
                        
                        processed_rows.append(tuple(current_row))

                    try:
                        cursor.executemany(insert_sql, processed_rows)
                        logger.info(f"Entry {entry_index}: Successfully inserted {len(processed_rows)} rows of data.")
                    except Exception as e:
                        logger.error(f"Entry {entry_index}: Data insertion failed: {e}")
                else:
                    logger.error(f"Entry {entry_index}: Table structure created successfully but no column information")

            conn.commit()
            return db_path, predicted_ddl, PipelineStatus.SUCCESS

        except sqlite3.Error as e:
            logger.error(f"Entry {entry_index}: DB connection or commit failed: {e}")
            return None, None, PipelineStatus.P1_UNKNOWN_ERROR
        finally:
            if conn: conn.close()

    def build_gold_db(
            self, 
            gold_schema_map: Dict[str, str], # {"frpm": "CREATE TABLE...", ...}
            insert_data_map: Dict[str, List[str]], # {"frpm": ["INSERT...", ...]}
            db_id: str
        ) -> str:
            subdir_name = f"{db_id}_gold"
            file_name = f"{db_id}_gold.sqlite"
            db_dir = os.path.join(self.output_dir, subdir_name)
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, file_name)
            if os.path.exists(db_path):
                try: os.remove(db_path)
                except OSError: pass

            conn = None
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                for table_name, ddl in gold_schema_map.items():
                    try:
                        cursor.execute(ddl)
                    except sqlite3.Error as e:
                        logger.warning(f"Gold DB table creation failed ({table_name}): {e}")

                for table_name, inserts in insert_data_map.items():
                    for sql in inserts:
                        try:
                            cursor.execute(sql)
                        except sqlite3.Error:
                            pass
                
                conn.commit()
                return db_path

            except Exception as e:
                logger.error(f"Failed to build Gold DB {db_id}: {e}")
                return None
            finally:
                if conn: conn.close()