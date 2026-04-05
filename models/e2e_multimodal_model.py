import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI

from utils.prompts import build_e2e_prompt
from utils.constants import PipelineStatus
from models.api_multimodal_model import load_image_bytes, slice_image, upload_to_oss_or_base64, slice_image_absolute_1to1

logger = logging.getLogger('YSHLogger')

class E2EMultimodalModel:
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.api_key = eval_config.e2e_api_key
        self.base_url = eval_config.e2e_api_endpoint
        self.model_name = eval_config.e2e_model_name
        logger.info(f"[E2EModel] Initializing: {self.model_name}")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=1200.0)

    def _sanitize_table_name(self, raw_name: str) -> str:
        name = raw_name.lower().strip()
        name = re.sub(r'\.(csv|xlsx|xls|json|txt|pdf|png|jpg|jpeg|md|html)$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[\s\-]+', '_', name.strip())
        name = re.sub(r'[^a-z0-9_]', '', name)
        return name

    def recognize_and_generate_sql(self, image_paths: List[str], question: str) -> Tuple[Optional[Dict], str, PipelineStatus]:
        """
        End-to-End Core Processing Logic: 
            Takes images and questions as input, and outputs table dictionaries and SQL.
        """
        try:
            content_list = []
            total_images_processed = 0

            for img_path in image_paths:
                original_bytes = load_image_bytes(img_path)
                if not original_bytes: continue
                if "internvl" in  self.model_name.lower() or 'llava' in self.model_name.lower():
                    logger.info(f"[E2EModel] {self.model_name} detected; skipping slicing and using original image.")
                    url = upload_to_oss_or_base64(original_bytes, self.eval_config)
                    content_list.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                    total_images_processed += 1
                elif "gpt" in  self.model_name.lower():
                    image_slices = slice_image_absolute_1to1(original_bytes, self.eval_config)
                    for img_slice in image_slices:
                        url = upload_to_oss_or_base64(img_slice, self.eval_config)
                        content_list.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                        total_images_processed += 1
                else:
                    image_slices = slice_image(original_bytes, self.eval_config)
                    for img_slice in image_slices:
                        url = upload_to_oss_or_base64(img_slice, self.eval_config)
                        content_list.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                        total_images_processed += 1

            if total_images_processed == 0:
                return None, "", PipelineStatus.DATA_IMAGES_ERROR

            user_prompt = build_e2e_prompt(question)
            if "InternVL".lower() in self.model_name.lower():
                logger.info("[InternVL] Using Tinking prompt.")
                system_prompt = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()
            elif "gpt" in self.model_name.lower():
                logger.info("[GPT] Using GPT prompt.")
                user_prompt += (
                    "You are a superhuman Data Extraction Engine evaluating sequential high-resolution image slices of a single massive database table.\n\n"
                    "CRITICAL EXTRACTION PROTOCOLS:\n"
                    "1. DE-DUPLICATION (OVERLAP AWARENESS): Because the images are sliced to preserve resolution, adjacent images have a visual overlap (e.g., the bottom rows of Image 1 are identical to the top rows of Image 2). "
                    "You MUST meticulously compare the boundary rows and perfectly DE-DUPLICATE them. Never output the same row twice.\n"
                    "2. SPATIAL ALIGNMENT (ANTI-SHIFTING): Trace the vertical alignment of the columns with your eyes. If a cell is visually blank or missing data, you MUST insert a pure JSON `null` (without quotes) in that exact column position. "
                    "DO NOT shift data from the right columns to the left to fill the gap. A misaligned column is a critical failure.\n"
                    "3. ABSOLUTE FIDELITY: Do not try to 'fix' typos, format dates, or add auto-incrementing ID columns (like row_index). Output only the explicit data shown.\n"
                    "4. MERGE STRATEGY: Maintain the exact schema derived from the header in the first image, and append all unique rows from all slices into ONE single JSON array."
                )
                system_prompt = "You are an expert Data Scientist and SQL Developer specialized in extracting data from table images and generating accurate SQL queries."
            else:
                system_prompt = "You are an expert Data Scientist and SQL Developer specialized in extracting data from table images and generating accurate SQL queries."

            content_list.append({"type": "text", "text": user_prompt})

            MAX_RETRIES = 3
            content = ""
            pred_sql = ""
            raw_data = None
            raw_tables = {}
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"[E2EModel] Starting attempt {attempt + 1} of {MAX_RETRIES}...")
                    if "InternVL".lower() in self.model_name.lower():
                        logger.info("[InternVL] Using 'temperature=0.6' config.")
                        api_params = dict(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": content_list},
                            ],
                            temperature=0.6,
                            max_tokens=self.eval_config.max_tokens,
                            seed=self.eval_config.seed,
                            # response_format={"type": "json_object"}
                        )
                    elif self.model_name == 'gemini-2.5-flash':
                        logger.info("[gemini-2.5-flash] reasoning_effort='none'")
                        api_params = dict(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": content_list},
                            ],
                            temperature=self.eval_config.temperature,
                            top_p=self.eval_config.top_p,
                            max_tokens=self.eval_config.max_tokens,
                            seed=self.eval_config.seed,
                            reasoning_effort="none",
                            response_format={"type": "json_object"}
                        )
                    else:
                        api_params = dict(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": content_list},
                            ],
                            temperature=self.eval_config.temperature,
                            top_p=self.eval_config.top_p,
                            max_tokens=self.eval_config.max_tokens,
                            seed=self.eval_config.seed,
                            response_format={"type": "json_object"}
                        )
                    
                    
                    is_stream = True if getattr(self.eval_config, 'level', '') != 'easy' else False
                    response = self.client.chat.completions.create(
                        **api_params,
                        stream=is_stream
                    )
                    content = ""
                    if not is_stream:
                        content = response.choices[0].message.content
                    else:
                        logger.info(f"[E2EModel] Receiving data stream (Level: {getattr(self.eval_config, 'level', 'unknown')})...")
                        collected_chunks = []
                        
                        for chunk in response:
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    collected_chunks.append(delta.content)
                        content = "".join(collected_chunks)
                        logger.info(f"[E2EModel] Streaming complete. Total character length: {len(content)}")

                    json_str = ""
                    json_blocks = re.findall(r'```(?:json)\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)

                    if json_blocks:
                        json_str = json_blocks[0].strip()
                    elif "{" in content:
                        match = re.search(r'\{.*\}', content, re.DOTALL)
                        if match: json_str = match.group(0)
                    else:
                        json_str = content.strip()

                    json_str_fixed = re.sub(r'([\[,:]\s*)(0\d+)(?=[,\]}])', r'\1"\2"', json_str)
                    
                    raw_data = json.loads(json_str_fixed)
                    
                    pred_sql = raw_data.get("sql", "").strip()
                    raw_tables = raw_data.get("tables", {})
                    
                    if not pred_sql and 'SELECT ' in content.upper():
                        match = re.search(r'(SELECT\s+.*?;)', content, re.DOTALL | re.IGNORECASE)
                        if match: pred_sql = match.group(1).strip()
                    
                    break 
                    
                except Exception as e:
                    logger.warning(f"[E2EModel] Parsing failed on attempt {attempt + 1}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        import time
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"[E2EModel] Retries exhausted. Final content preview: {content}")
                        return None, pred_sql, PipelineStatus.P1_MODEL_OUTPUT_INVALID

            cleaned_data = {}
            for raw_key, table_content in raw_tables.items():
                clean_key = self._sanitize_table_name(raw_key)
                
                if clean_key in cleaned_data: 
                    clean_key = f"{clean_key}_part"
                
                cleaned_data[clean_key] = table_content
                
            return cleaned_data, pred_sql, PipelineStatus.SUCCESS

        except Exception as e:
            logger.error(f"[E2EModel] Processing exception: {e}")
            return None, "", PipelineStatus.P1_UNKNOWN_ERROR