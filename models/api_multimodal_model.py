import oss2
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Any, Optional
import os, json, base64, hashlib, re, logging, mimetypes
from utils.prompts import build_visual_extraction_prompt
from utils.constants import PipelineStatus
logger = logging.getLogger('YSHLogger')


def load_image_bytes(image_path: str) -> Optional[bytes]:
    """
    Read image data. By default, the original image is read directly to ensure maximum clarity.
    """
    try:
        with open(image_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Image read error [{os.path.basename(image_path)}]: {e}")
        return None

def encode_to_jpeg(img: Image.Image, quality: int = 100) -> bytes:
    """
    Convert the PIL Image object into a high-quality JPEG byte stream.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    with BytesIO() as buffer:
        img.save(buffer, format="JPEG", quality=quality, subsampling=0)
        return buffer.getvalue()

def slice_image(image_bytes: bytes, config) -> List[bytes]:
    """
    daptive Image Processor:
        1.Restricts total pixels and width to avoid API-induced blur.
        2.Performs vertical splitting only for ultra-long images.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            w, h = img.size
            ratio = h / w
            pixels_mp = (w * h) / 1e6

            # --- Downscaling ---
            if w > config.max_width or pixels_mp > 10.0:
                # scale = config.max_width / w
                scale = min(1.0, config.max_width / w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.info(f"[Rescale] Original size is: {w}x{h}")
                w, h = new_w, new_h
                ratio = h / w
                logger.info(f"[Rescale] Resized to: {w}x{h} (Ratio: {ratio:.2f})")

            # --- Conditional Slicing---
            should_slice = h > config.max_height and ratio > 3.0

            if not should_slice:
                logger.debug(f"[SkipSlice] Meets API direct processing criteria.")
                return [encode_to_jpeg(img)]

            slices = []
            top = 0
            while top < h:
                bottom = min(top + config.max_height, h)
                
                if (h - bottom) < (config.max_height * 0.25) and len(slices) > 0:
                    bottom = h
                
                crop = img.crop((0, top, w, bottom))
                slices.append(encode_to_jpeg(crop))
                
                if bottom == h: break
                top += (config.max_height - config.overlap)

            logger.info(f"[Slice] Extreme aspect ratio detected: Splitting into {len(slices)} segments")
            return slices

    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return [image_bytes]

def slice_image_for_GPT(image_bytes: bytes, config) -> List[bytes]:
    GPT_MAX_SHORT_SIDE = 768
    GPT_MAX_LONG_SIDE = 2048
    
    OVERLAP = getattr(config, 'overlap', 150)
    logger.info("[GPT] Using 'slice_image_for_GPT' method.")
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            w, h = img.size
            logger.info(f"[GPT Slicer] Original size is: {w}x{h}")
          
            if w > GPT_MAX_SHORT_SIDE and h >= w: 
                scale = GPT_MAX_SHORT_SIDE / w
                new_w = GPT_MAX_SHORT_SIDE
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.info(f"[GPT Slicer] High-quality local rescaling for long image to: {new_w}x{new_h}")
                w, h = new_w, new_h
                
            elif h > GPT_MAX_SHORT_SIDE and w > h:
                scale = GPT_MAX_SHORT_SIDE / h
                new_h = GPT_MAX_SHORT_SIDE
                new_w = int(w * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.info(f"[GPT Slicer] High-quality local rescaling for wide image to: {new_w}x{new_h}")
                w, h = new_w, new_h

            if h <= GPT_MAX_LONG_SIDE:
                logger.debug(f"[GPT Slicer] Dimensions {w}x{h} within API safety limits; passing original image")
                return [encode_to_jpeg(img)]

            slices = []
            top = 0
            
            while top < h:
                bottom = min(top + GPT_MAX_LONG_SIDE, h)
                if (h - bottom) < (GPT_MAX_LONG_SIDE * 0.15) and len(slices) > 0:
                    bottom = h
                    top = max(0, h - GPT_MAX_LONG_SIDE)
                crop = img.crop((0, top, w, bottom))
                slices.append(encode_to_jpeg(crop))
                
                if bottom >= h: 
                    break
                    
                top += (GPT_MAX_LONG_SIDE - OVERLAP)
            logger.info(f"[GPT Slicer] Long image height exceeds limit; losslessly split into {len(slices)} segments.")
            return slices

    except Exception as e:
        logger.error(f"[GPT Slicer] Image slicing error: {e}")
        return [image_bytes]

def slice_image_absolute_1to1(image_bytes: bytes, config) -> List[bytes]:
    """
    [1:1 Slicing] Zero-Scale 2D Grid Slicer (for GPT)
    """
    GPT_MAX_W = 768
    GPT_MAX_H = 2048
    
    overlap_y = getattr(config, 'overlap', 150)
    overlap_x = 100
    
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            w, h = img.size
            logger.info(f"[2D Slicer] Orignal size: {w}x{h}")

            if w <= GPT_MAX_W and h <= GPT_MAX_H:
                logger.debug(f"[2D Slicer] Dimensions safe; no slicing required.")
                return [encode_to_jpeg(img)]

            slices = []
            top = 0
            row_idx = 0
            
            while top < h:
                bottom = min(top + GPT_MAX_H, h)
                if (h - bottom) < (GPT_MAX_H * 0.15) and top > 0:
                    bottom = h
                    top = max(0, h - GPT_MAX_H)

                left = 0
                col_idx = 0
                
                while left < w:
                    right = min(left + GPT_MAX_W, w)
                    if (w - right) < (GPT_MAX_W * 0.15) and left > 0:
                        right = w
                        left = max(0, w - GPT_MAX_W)

                    crop = img.crop((left, top, right, bottom))
                    slices.append(encode_to_jpeg(crop))
                    logger.debug(f"  -> Generating grid [{row_idx},{col_idx}], Size: {right-left}x{bottom-top}")
                    if right >= w: break
                    left += (GPT_MAX_W - overlap_x) 
                    col_idx += 1

                if bottom >= h: break
                top += (GPT_MAX_H - overlap_y)
                row_idx += 1

            logger.info(f"[2D Slicer] Absolute lossless slicing complete! Generated {row_idx+1} rows x {col_idx+1} columns, totaling {len(slices)} grid fragments.")
            return slices

    except Exception as e:
        logger.error(f"[2D Slicer] Image slicing error: {e}")
        return [image_bytes]

_oss_bucket = None
def get_oss_bucket(eval_config):
    global _oss_bucket
    if _oss_bucket is None:
        try:
            if not eval_config.use_oss: return None
            auth = oss2.Auth(eval_config.oss_access_key_id, eval_config.oss_access_key_secret)
            _oss_bucket = oss2.Bucket(auth, eval_config.oss_endpoint, eval_config.oss_bucket_name)
        except Exception as e:
            logger.error(f"Failed to initialize OSS Bucket: {e}")
            return None
    return _oss_bucket

def upload_to_oss_or_base64(image_bytes: bytes, eval_config) -> str:
    """
    Attempts to upload to OSS; falls back to Base64 if upload fails or is not configured.
    Returns: The complete URL (e.g., http... or data:image...)
    """
    bucket = get_oss_bucket(eval_config)
    if bucket:
        try:
            file_hash = hashlib.md5(image_bytes).hexdigest()
            object_key = f"{eval_config.oss_dir_prefix.strip('/')}/{file_hash}.jpg"

            if not bucket.object_exists(object_key):
                bucket.put_object(object_key, image_bytes)

            endpoint_domain = eval_config.oss_endpoint.replace("http://", "").replace("https://", "")
            return f"https://{eval_config.oss_bucket_name}.{endpoint_domain}/{object_key}"
        except Exception as e:
            logger.warning(f"OSS upload failed; falling back to Base64: {e}")
    
    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

# ==========================================
# 3. Main Model Class
# ==========================================
class APIMultimodalModel:
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.api_key = eval_config.visual_model_api_key
        self.base_url = eval_config.visual_model_base_url
        self.model_name = eval_config.visual_model
        logger.info(f"[VisualModel] Initializing: {self.model_name}")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=1200.0)

    def _sanitize_table_name(self, raw_name: str) -> str:
        """
        Table Name Cleaning Rules:

            1.Remove file extensions (e.g., .csv, .xlsx, .pdf).

            2.Convert to lowercase.

            3.Trim leading and trailing whitespace.

            4.Replace all non-alphanumeric characters (except underscores) with underscores.
        """
        if not raw_name:
            return "table_unknown"
        name = raw_name.lower().strip()
        name = re.sub(r'\.(csv|xlsx|xls|json|txt|pdf|png|jpg|jpeg|md|html)$', '', name, flags=re.IGNORECASE)
        name = name.strip()
        name = re.sub(r'[\s\-]+', '_', name)
        name = re.sub(r'[^a-z0-9_]', '', name)
        
        return name

    
    
    def recognize_table_data(self, image_paths: List[str]) -> Optional[Dict[str, Dict[str, List[Any]]]]:
        """
        Core Recognition Logic:
        1. Receive a list of image paths.
        2. Process each image: Read -> Adaptive Slicing -> Convert to URL.
        3. Consolidate all slices/images into a single content_list.
        4. Execute a batch API call.

        return: dict
        {
            "employees": {
                "schema_sql": "CREATE TABLE employees (employee_id INTEGER, name TEXT, salary REAL, hire_date TEXT);",
                "headers": ["employee_id", "Name", "Salary", "Hire Date"],
                "rows": [
                ]
            },
            "departments": {
                "schema_sql": "CREATE TABLE departments (dept_id INTEGER, dept_name TEXT);",
                "headers": ["ID", "Department Name"],
                "rows": [
                ]
            }
        }
        """
        try:
            content_list = []
            total_images_processed = 0
            # =================================================================
            # [Debug Feature] 1. Initialize Root Directory for Current Request (Level 1)
            # Directory Format: debug_slices/Request_YYYYMMDD_HHMMSS_ffffff
            # =================================================================
            req_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_req_dir = os.path.join(self.eval_config.run_dir, "debug_slices", f"Request_{req_timestamp}")
            
            ENABLE_DEBUG_SAVE = False 
            
            if ENABLE_DEBUG_SAVE:
                os.makedirs(debug_req_dir, exist_ok=True)
                logger.info(f"[Debug] Slices saved to: {debug_req_dir}")
            # =================================================================

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
                # [Debug Feature] 2. Setup Subdirectory for Each Image (Level 2) and Save Slices
                # Directory Format: debug_slices/Request_.../original_filename/
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
                            logger.warning(f"Failed to save debug slices: {e}")
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
                return None, PipelineStatus.DATA_IMAGES_ERROR

            user_prompt = build_visual_extraction_prompt()
            
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
            system_prompt = "You are an expert Data Scientist and SQL Developer specialized in extracting data from table images using SQLite. Do not truncate data. Extract all rows visible."

            MAX_RETRIES = 3
            response = None
            raw_data = None
            content = None
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"[VisualModel] Starting API call (Attempt {attempt + 1} of {MAX_RETRIES})...")
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
                    # print(response)
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
                    
                    raw_data = None
                    raw_data = json.loads(json_str_fixed)
                    logger.info("[VisualModel] JSON parsed successfully.")
                    break
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
                        logger.error("[VisualModel] Maximum retries exceeded; task failed.")
                        return None, PipelineStatus.P1_UNKNOWN_ERROR        
        except Exception as e:
            logger.error(e)
            return None, PipelineStatus.P1_UNKNOWN_ERROR

        if not raw_data:
            return None, PipelineStatus.P1_MODEL_OUTPUT_EMPTY

        if not isinstance(raw_data, dict):
            logger.error(f"[VisualModel] Parsed result is not a dictionary or is empty. Content preview: {content}")
            logger.info(content)
            logger.info(image_paths)
            return None, PipelineStatus.P1_MODEL_OUTPUT_INVALID


        cleaned_data = {}
        for raw_key, table_content in raw_data.items():
            clean_key = self._sanitize_table_name(raw_key)
            if clean_key in cleaned_data:
                clean_key = f"{clean_key}_part"

            if "schema_sql" not in table_content:
                logger.warning(f"[VisualModel] Table '{clean_key}' is missing 'schema_sql'. DatabaseBuilder will use fallback.")
            
            cleaned_data[clean_key] = table_content
        
        return cleaned_data, PipelineStatus.SUCCESS













