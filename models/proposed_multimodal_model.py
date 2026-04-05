import os
import re
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI
from datetime import datetime
from utils.prompts import build_proposed_extraction_prompt
from utils.constants import PipelineStatus
from models.api_multimodal_model import load_image_bytes, slice_image, upload_to_oss_or_base64, slice_image_absolute_1to1

logger = logging.getLogger('YSHLogger')

class ProposedMultimodalModel:
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.api_key = eval_config.visual_model_api_key
        self.base_url = eval_config.visual_model_base_url
        self.model_name = eval_config.visual_model
        logger.info(f"[ProposedModel] Initializing: {self.model_name}")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=1200.0)

    def _sanitize_table_name(self, raw_name: str) -> str:
        name = raw_name.lower().strip()
        name = re.sub(r'\.(csv|xlsx|xls|json|txt|pdf|png|jpg|jpeg|md|html)$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[\s\-]+', '_', name.strip())
        name = re.sub(r'[^a-z0-9_]', '', name)
        return name

    def recognize_table_and_hints(self, question, image_paths: List[str]) -> Tuple[Optional[Dict], str, PipelineStatus]:
        """
        Core Recognition Logic: 
        Returns the parsed table dictionary, extracted visual prompt string, and execution status.
        """
        try:
            content_list = []
            total_images_processed = 0
            # =================================================================
            # [Debug] 1. Prepare root debug directory for current request (Level 1)
            # Directory format: debug_slices/Request_YYYYMMDD_HHMMSS_ffffff
            # =================================================================
            req_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_req_dir = os.path.join(self.eval_config.run_dir, "debug_slices", f"Request_{req_timestamp}")
            
            ENABLE_DEBUG_SAVE = False 
            
            if ENABLE_DEBUG_SAVE:
                os.makedirs(debug_req_dir, exist_ok=True)
                logger.info(f"[Debug] Slice storage directory: {debug_req_dir}")

            for img_path in image_paths:
                original_bytes = load_image_bytes(img_path)
                if not original_bytes: 
                    logger.error(f"[VisualModel] Skipping unreadable image: {img_path}")
                    continue
                if 'GPT'.lower() in self.model_name.lower():
                    image_slices = slice_image_absolute_1to1(original_bytes, self.eval_config)
                else:
                    image_slices = slice_image(original_bytes, self.eval_config)
                # =================================================================
                # [Debug] 2. Prepare subdirectories for each image (Level 2) and save slices
                # Directory format: debug_slices/Request_.../original_filename/
                # =================================================================
                if ENABLE_DEBUG_SAVE:
                    img_name = os.path.basename(img_path)
                    safe_img_name = re.sub(r'[^\w\-_\.]', '_', img_name)
                    debug_img_dir = os.path.join(debug_req_dir, safe_img_name)
                    os.makedirs(debug_img_dir, exist_ok=True)

                    for idx, slice_bytes in enumerate(image_slices):
                        slice_filename = f"slice_{idx:02d}.jpg"
                        save_path = os.path.join(debug_img_dir, slice_filename)
                        try:
                            with open(save_path, "wb") as f:
                                f.write(slice_bytes)
                        except Exception as e:
                            logger.warning(f"Failed to save debug slice: {e}")
                # =================================================================
                for img_slice in image_slices:
                    url = upload_to_oss_or_base64(img_slice, self.eval_config)
                    content_list.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": url,
                            "detail": "high"
                        }
                    })
                    total_images_processed += 1

            if total_images_processed == 0:
                logger.error("[VisualModel] No valid images were processed.")
                return None, "", PipelineStatus.DATA_IMAGES_ERROR

            # 2. Construct prompt including Visual Hints extraction instructions
            user_prompt = build_proposed_extraction_prompt(question=question)
            if 'GPT'.lower()  in self.model_name.lower():
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
            else:
                user_prompt += (
                    "Please extract the database schema and data from the provided image(s). "
                    "Do not truncate data. Extract all rows visible."
                    "If multiple image parts are provided, treat them as continuous vertical slices of the same table/document. "
                    "Merge them into a coherent structure and handle any repeating headers automatically."
                )
            content_list.append({"type": "text", "text": user_prompt})

            if "InternVL" in self.model_name:
                system_prompt = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()
            else:
                system_prompt = "You are an expert Data Extraction AI. You strictly follow the required JSON structure."

            MAX_RETRIES = 3
            response = None
            raw_data = None
            visual_hints = None
            content = None
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"[VisualModel] Starting API call {attempt + 1} of {MAX_RETRIES}...")
                    if "InternVL" in self.model_name:
                        logger.info("[InternVL] Using 'temperture=0.6, do_sample=True' when thinking module.")
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": content_list},
                            ],
                            temperature=0.6,
                            max_tokens=self.eval_config.max_tokens,
                            seed=self.eval_config.seed,
                            stream=True if self.eval_config.level != 'easy' else False,
                            response_format={"type": "json_object"}
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": content_list},
                            ],
                            temperature=self.eval_config.temperature,
                            top_p=self.eval_config.top_p,
                            frequency_penalty=self.eval_config.frequency_penalty,
                            presence_penalty=self.eval_config.presence_penalty,
                            max_tokens=self.eval_config.max_tokens,
                            seed=self.eval_config.seed,
                            stream=True if self.eval_config.level != 'easy' else False,
                            response_format={"type": "json_object"}
                        )

                    content = ""
                    if self.eval_config.level == 'easy':
                        content = response.choices[0].message.content
                    else:
                        logger.info(f"[VisualModel] Receiving data stream (Level: {self.eval_config.level})...")
                        collected_chunks = []
                        for chunk in response:
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if delta.content:
                                    collected_chunks.append(delta.content)
                        content = "".join(collected_chunks)
                        logger.info(f"[VisualModel] Streaming complete. Total character length: {len(content)}")
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        json_str = content.replace("```", "").strip()
                    else:
                        json_str = content.strip()

                    json_str_fixed = re.sub(r'([\[,:]\s*)(0\d+)(?=[,\]}])', r'\1"\2"', json_str)
                    
                    raw_data = json.loads(json_str_fixed)
                    visual_hints = raw_data.get("visual_hints", "")
                    raw_tables = raw_data.get("tables", {})
                except Exception as e:
                    logger.info("=====[DEBUG]=====")
                    logger.info(response)
                    logger.info("=====[DEBUG]=====")
                    logger.info(content)
                    logger.info("=====[DEBUG]=====")
                    logger.info(raw_data)
                    logger.info("=====[DEBUG]=====")
                    logger.error(f"[VisualModel] Exception occurred on attempt {attempt + 1}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        import time
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"[ProposedModel] Retries exhausted: {e}")
                        return None, "", PipelineStatus.P1_UNKNOWN_ERROR

            
            if not raw_data:
                return None, "", PipelineStatus.P1_MODEL_OUTPUT_EMPTY
            
            # =========================================================
            # [Debug Module] Handling cases where raw_data 
            # is parsed successfully but is not a dictionary
            # =========================================================
            if not isinstance(raw_data, dict):
                logger.error(f"[VisualModel] Parse result type error: expected dict, but got {type(raw_data)}")
                
                if self.eval_config.level == 'easy':
                    if hasattr(response, 'usage') and response.usage:
                        logger.info(f"[Debug Token Usage] Prompt: {response.usage.prompt_tokens}, "
                                    f"Completion: {response.usage.completion_tokens}, "
                                    f"Total: {response.usage.total_tokens}")
                    else:
                        logger.info("[Debug Token Usage] Token usage information not found in the response.")
                else:
                    logger.info("[Debug Token Usage] Token usage information not found in the response.")

                if self.eval_config.level == 'easy':
                    try:
                        if hasattr(response, 'model_dump'):
                            logger.info("[Debug Response Meta]\n" + json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
                        elif hasattr(response, 'to_dict'):
                            logger.info("[Debug Response Meta]\n" + json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
                        else:
                            logger.info(f"[Debug Raw Response] {response}")
                    except Exception as e:
                        logger.warning(f"[Debug] Failed to serialize Response object: {e}")

                content_len = len(content) if content else 0
                logger.error(f"[Debug] Extracted raw content length: {content_len} characters")
                
                if content_len > 0:
                    logger.error(f"[Debug] Content preview:\n{content}")
                else:
                    logger.error("[Debug] Content is empty!")

                return None, "", PipelineStatus.P1_MODEL_OUTPUT_INVALID
            # =========================================================

            cleaned_tables = {}
            for raw_key, table_content in raw_tables.items():
                clean_key = self._sanitize_table_name(raw_key)
                if clean_key in cleaned_tables:
                    clean_key = f"{clean_key}_part"
                cleaned_tables[clean_key] = table_content

            return cleaned_tables, visual_hints, PipelineStatus.SUCCESS

        except Exception as e:
            logger.error(f"[ProposedModel] Exception during processing: {e}")
            return None, "", PipelineStatus.P1_UNKNOWN_ERROR