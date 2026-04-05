import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
logger = logging.getLogger('YSHLogger')

class VisualCacheManager:
    def __init__(self, cache_root_dir: str):
        self.cache_root_dir = cache_root_dir
        os.makedirs(self.cache_root_dir, exist_ok=True)

    def _normalize_meta(self, meta: Dict[str, Any]) -> str:
        """Convert metadata to a stable string for comparison (sort keys)."""
        return json.dumps(meta, sort_keys=True)

    def get_cached_result(self, entry_id: str, expected_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cache and perform strict metadata validation.
        """
        file_name = f"entry_{entry_id}.json"
        file_path = os.path.join(self.cache_root_dir, file_name)

        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cached_obj = json.load(f)
            
            if "meta" not in cached_obj or "content" not in cached_obj:
                logger.warning(f"Cache {entry_id} format obsolete. Ignoring.")
                return None

            stored_meta = cached_obj["meta"]
            
            if self._normalize_meta(stored_meta) != self._normalize_meta(expected_metadata):
                # logger.info(f"Cache {entry_id} metadata mismatch. \nStored: {stored_meta}\nExpected: {expected_metadata}")
                return None
            
            return cached_obj["content"]

        except Exception as e:
            logger.warning(f"Failed to read cache {file_path}: {e}")
            return None

    def save_result(self, entry_id: str, result_data: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Save results with metadata.
        """
        if not result_data: return

        file_name = f"entry_{entry_id}.json"
        file_path = os.path.join(self.cache_root_dir, file_name)

        envelope = {
            "meta": metadata,
            "content": result_data
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(envelope, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write cache {file_path}: {e}")

class Text2SQLCacheManager:
    """
    Phase 2 Cache Manager
    Function: Fingerprint validation based on Input Data Hash and Config Hash.
    """
    def __init__(self, input_path: str, output_path: str, model_name: str):
        """
        Args:
            input_path: Path to generated_dev_tables.json from Phase 1 (used for input hash calculation).
            output_path: Path to final_predictions.json from Phase 2 (used for result checking).
            model_name: Name of the current SOTA model (used to distinguish between different models).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.meta_path = str(Path(output_path).with_suffix('.meta.json'))

    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate the MD5 hash of a file."""
        if not os.path.exists(filepath):
            return ""
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            return file_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {filepath}: {e}")
            return ""

    def get_current_fingerprint(self) -> dict:
        """Generate the current runtime fingerprint."""
        input_hash = self._calculate_file_hash(self.input_path)
        return {
            "input_data_hash": input_hash,
            "model_name": self.model_name,
        }

    def check_cache(self, data_num) -> bool:
        """
        [Core Logic] Check if the cache is valid.

        Conditions:
        1. Output file exists and is non-empty.
        2. Metadata file exists.
        3. Fingerprint in metadata matches the current fingerprint exactly.
        """
        if not os.path.exists(self.output_path):
            return False
        
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict) or len(data.keys()) != data_num:
                logger.info("[Text2SQL Cache] Cache Miss: Output file is invalid.")
                return False
        except Exception:
            return False

        if not os.path.exists(self.meta_path):
            logger.info("[Text2SQL Cache] Cache Miss: Metadata file not found (First run with new logic?).")
            return False

        try:
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                stored_meta = json.load(f)
            
            current_fingerprint = self.get_current_fingerprint()
            
            if stored_meta.get("input_data_hash") != current_fingerprint["input_data_hash"]:
                logger.info("[Text2SQL Cache] Cache Miss: Input data (Phase 1 result) has changed.")
                return False
            
            if stored_meta.get("model_name") != current_fingerprint["model_name"]:
                logger.info(f"[Text2SQL Cache] Cache Miss: Model changed ({stored_meta.get('model_name')} -> {current_fingerprint['model_name']}).")
                return False

            logger.info(f"[Text2SQL Cache] HIT! Fingerprint matches. Using cached results from {os.path.basename(self.output_path)}")
            return True

        except Exception as e:
            logger.warning(f"[Text2SQL Cache] Meta check failed: {e}")
            return False

    def save_metadata(self):
        """Save the current fingerprint information after successful inference."""
        try:
            meta = self.get_current_fingerprint()
            meta["timestamp"] = __import__("datetime").datetime.now().isoformat()
            
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=4)
            logger.info(f"[Text2SQL Cache] Metadata saved to {os.path.basename(self.meta_path)}")
        except Exception as e:
            logger.error(f"[Text2SQL Cache] Failed to save metadata: {e}")