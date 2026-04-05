import re
import sqlite3
import logging
from .constants import PipelineStatus
from typing import Optional, Dict, Any, List
logger = logging.getLogger('YSHLogger')

class PipelineDataConstructor:
    """
    Responsible for converting structured data extracted by the visual model into BIRD dataset format (dev_tables.json, dev.json). 
    Preserve the data types supported by BIRD (integer, real, text, date, etc.).
    """
    def __init__(self):
        pass

    def _parse_types_from_ddl(self, t_name: str, ddl: str, headers: List[str]) -> List[str]:
        """
        Parse the column types in a DDL.
            param t_name: The expected table name (used for prioritizing the query)
            param ddl: CREATE TABLE statement
            param headers: List of table headers (used for final alignment)
        """
        if not ddl: 
            return ['text'] * len(headers)

        type_mapping = {
            'INTEGER': 'integer', 'INT': 'integer', 'TINYINT': 'integer', 'SMALLINT': 'integer', 'BIGINT': 'integer', 'YEAR': 'integer',
            'REAL': 'real', 'FLOAT': 'real', 'DOUBLE': 'real', 'DECIMAL': 'real', 'NUMERIC': 'real',
            'DATE': 'date', 'DATETIME': 'date', 'TIMESTAMP': 'date', 'TIME': 'date',
            'TEXT': 'text', 'VARCHAR': 'text', 'CHAR': 'text', 'CLOB': 'text', 'STRING': 'text', 'BOOLEAN': 'integer'
        }

        conn = None
        try:
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            
            cursor.execute(ddl)
            
            cursor.execute(f"PRAGMA table_info('{t_name}')")
            columns_info = cursor.fetchall()
            
            if not columns_info:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                row = cursor.fetchone()
                if row:
                    actual_table_name = row[0]
                    logger.warning(f"[Schema Parser] Table name does not match: expected '{t_name}', DDL generated '{actual_table_name}'. Using the actual table name for parsing.")
                    cursor.execute(f"PRAGMA table_info('{actual_table_name}')")
                    columns_info = cursor.fetchall()
                else:
                    logger.warning(f"[Schema Parser] DDL execution seems successful but no tables were found. DDL: {ddl[:50]}...")
                    return ['text'] * len(headers)


            extracted_types = []
            for col_info in columns_info:
                raw_type = col_info[2].upper()
                simple_type = raw_type.split('(')[0].strip()
                bird_type = type_mapping.get(simple_type, 'text')
                extracted_types.append(bird_type)
            
            return extracted_types

        except sqlite3.Error as e:
            logger.warning(f"[Schema Parser] SQLite parsing exception: {e}. DDL: {ddl}")
            return ['text'] * len(headers)
        finally:
            if conn: conn.close()

    def construct_table_entry(self, db_id: str, visual_data: Dict[str, Any]):
        """
        Construct a single entry in dev_tables.json.
        """
        table_names_original = []
        table_names = [] 
        
        column_names_original = [[-1, "*"]]
        column_names = [[-1, "*"]]
        column_types = ["text"]
        
        data_map = visual_data
        for table_idx, (table_name_key, content) in enumerate(data_map.items()):
            if not isinstance(content, dict):
                logger.error(f"Skipping invalid table data for '{table_name_key}' in Item {db_id} (Type: {type(content)})")
                continue
            t_name = table_name_key
            headers = content.get('headers', [])
            schema_sql = content.get('schema_sql', "")
            if not headers:
                logger.error(f"Entry {db_id}: Table {t_name} is missing headers, skipping.")
                return False, None, PipelineStatus.P1_SCHEMA_CREATE_FAILED
            if not schema_sql:
                logger.error(f"Entry {db_id}: Table {t_name} is missing schema_sql, skipping.")
                return False, None, PipelineStatus.P1_SCHEMA_CREATE_FAILED

            ddl_types = self._parse_types_from_ddl(t_name, schema_sql, headers)
            
            if len(ddl_types) != len(headers):
                logger.error(f"Entry {db_id}: Table {t_name} column count mismatch. Headers({len(headers)}) vs DDL({len(ddl_types)})")
                logger.error(f"Entry {db_id}: DDL: {schema_sql}, Headers: {headers}")
                return False, None, PipelineStatus.P1_SCHEMA_CREATE_FAILED

            table_names_original.append(t_name)
            table_names.append(t_name) 
            
            for col_idx, header in enumerate(headers):
                h_name = str(header).strip()
                
                column_names_original.append([table_idx, h_name])
                column_names.append([table_idx, h_name])
                column_types.append(ddl_types[col_idx])

        return True, {
            "db_id": db_id,
            "table_names_original": table_names_original,
            "table_names": table_names,
            "column_names_original": column_names_original,
            "column_names": column_names,
            "column_types": column_types,
            "primary_keys": [], 
            "foreign_keys": []  
        }, PipelineStatus.SUCCESS

    def construct_dev_entry(self, new_db_id: str, original_item: Dict[str, Any]):
        """Construct dev.json entry"""
        assert int(new_db_id) == int(original_item.get("question_id"))
        return True, {
            "question_id": int(original_item.get("question_id")),
            "db_id": str(new_db_id), 
            "question": original_item.get("question"),
            "evidence": "", 
            "SQL": original_item.get("SQL", ""), 
            "difficulty": original_item.get("difficulty", "simple")
        }, PipelineStatus.SUCCESS

    
    def construct_dev_entry_with_hints(self, new_db_id: str, original_item: Dict[str, Any], visual_hints):
        """Construct dev.json entry with hints"""
        assert int(new_db_id) == int(original_item.get("question_id"))
        return True, {
            "question_id": int(original_item.get("question_id")),
            "db_id": str(new_db_id), 
            "question": original_item.get("question"),
            "evidence": str(visual_hints), 
            "SQL": original_item.get("SQL", ""), 
            "difficulty": original_item.get("difficulty", "simple")
        }, PipelineStatus.SUCCESS