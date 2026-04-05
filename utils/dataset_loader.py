import json
import logging
import os 
from typing import List, Dict, Any, Optional
from .config import PipelineConfig
logger = logging.getLogger('YSHLogger')

class EvaluationDatasetLoader:
    def __init__(
        self,
        eval_config: PipelineConfig,
        limit: Optional[int] = None
    ):
        self.visual_dev_path = eval_config.visual_dev_json_path
        self.visual_data_path = eval_config.visual_data_json_path
        self.limit = limit
        self.data = self._load_main_data()
        self.insert_data_map = self._load_insert_data()

    def _load_main_data(self) -> List[Dict[str, Any]]:
        """
        [
            {
                "question_id": 1383,
                "db_id": "student_club",
                "question": "State the name of students from Georgetown, South Carolina.",
                "evidence": "name of students means the full name; full name refers to first_name, last_name; Georgetown is a city; South Carolina is a state",
                "SQL": "SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T2.city = 'Georgetown' AND T2.state = 'South Carolina'",
                "difficulty": "simple",
                "image_paths": [
                    ....
                ],
                "gold_tables": {
                    "member": "CREATE TABLE \"member\" (\n  member_id TEXT CONSTRAINT member_pk PRIMARY KEY,\n  first_name TEXT,\n  last_name TEXT,\n  email TEXT,\n  position TEXT,\n  t_shirt_size TEXT,\n  phone TEXT,\n  zip INTEGER,\n  link_to_major TEXT\n)",
                    "zip_code": "CREATE TABLE zip_code (\n  zip_code INTEGER CONSTRAINT zip_code_pk PRIMARY KEY,\n  type TEXT,\n  city TEXT,\n  state TEXT\n)"
                }
            },
            ...
        ]
        """
        if not os.path.exists(self.visual_dev_path):
            logger.error(f"Evaluation dataset not found at path: '{self.visual_dev_path}'")
            return []
        
        try:
            with open(self.visual_dev_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if self.limit and isinstance(data, list):
                    logger.debug(f"DEBUG Mode: Loading first {self.limit} items only [Total: {len(data)}]")
                    data = data[:self.limit]
                logger.info(f"Successfully loaded {len(data)} evaluation items (visual_dev.json) from {self.visual_dev_path}.")
                return data
        except Exception as e:
            logger.error(f"Error while loading evaluation dataset (visual_dev.json) {self.visual_dev_path}: {e}", exc_info=True)
            return []

    def _load_insert_data(self) -> Dict[str, Any]:
        """
        {
            "1383": {
                "member": [
                    "INSERT INTO \"member\" (\"member_id\", \"first_name\", \"last_name\", \"email\", \"position\", \"t_shirt_size\", \"phone\", \"zip\", \"link_to_major\") VALUES ('recttfySfQnYb68u3', 'Annabella', 'Warren', 'annabella.warren@lpu.edu', 'Secretary', 'Large', '727-555-2732', 60047, 'rectez0Ce1okUhv8w');",
                    ...
                ],
                "zip_code": [
                    "INSERT INTO \"zip_code\" (\"zip_code\", \"type\", \"city\", \"state\") VALUES (29805, 'Standard', 'Aiken', 'South Carolina');",
                    ...
                ]
            },
            ....
        }
        """
        if not os.path.exists(self.visual_data_path):
            logger.error(f"INSERT data file (visual_data.json) not found: {self.visual_data_path}")
            return {}
        try:
            with open(self.visual_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded {len(data)} INSERT data (visual_data.json) entries from {self.visual_data_path}.")
                return data
        except Exception as e:
            logger.error(f"Error loading INSERT dataset (visual_data.json) at {self.visual_data_path}: {e}", exc_info=True)
            return {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if 0 <= index < len(self.data):
            return self.data[index]
        else:
            raise IndexError("Index out of range")

    def get_data(self) -> List[Dict[str, Any]]:
        return self.data

    def get_insert_data_map(self) -> Dict[str, Any]:
        return self.insert_data_map

    def get_gold_inserts(self, question_id: str) -> Dict[str, List[str]]:
        return self.insert_data_map.get(str(question_id), {})

    def get_gt_item_map(self) -> Dict[str, Dict]:
        item_map = {}
        for item in self.data:
            key = str(item.get('question_id'))
            item_map[key] = item
        return item_map