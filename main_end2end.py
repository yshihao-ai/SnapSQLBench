import os
import yaml
import json
import logging
import asyncio
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List
import sqlite3
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from utils.config import E2EConfig
from utils.constants import PipelineStatus
from utils.prompts import build_e2e_prompt
from utils.dataset_loader import EvaluationDatasetLoader
from utils.db_builder import DatabaseBuilder
from utils.pipeline_data_constructor import PipelineDataConstructor
from utils.cache_manager import VisualCacheManager
from utils.evaluator import Evaluator

from models.e2e_multimodal_model import E2EMultimodalModel

logger = logging.getLogger('YSHLogger')

async def process_e2e_phase(
    semaphore: asyncio.Semaphore,
    eval_config: E2EConfig,
    item: Dict,
    mm_model: E2EMultimodalModel,
    db_builder: DatabaseBuilder, 
    data_constructor: PipelineDataConstructor,
    visual_cache_manager: VisualCacheManager,
    global_prompt_hash: str,
    global_config_hash: str,
    skip_api_call: bool = False
):
    index = item.get('question_id')
    entry_id = str(index)
    question = item.get('question', '')
    raw_image_paths = item.get('image_paths', [])
    
    if not raw_image_paths:
        logger.error(f"Item {index}: No images found.")
        return False, index, PipelineStatus.DATA_IMAGES_ERROR, ""

    target_images = []
    for path in raw_image_paths:
        full_path = path if os.path.isabs(path) else os.path.join(eval_config.project_root, path)
        if os.path.exists(full_path):
            target_images.append(full_path)
        else:
            return False, index, PipelineStatus.DATA_IMAGES_ERROR, ""

    target_image_names = [os.path.basename(p) for p in target_images]
    target_image_names.sort()
    
    current_metadata = {
        "image_names": target_image_names,
        "level": eval_config.level,
        "model": eval_config.e2e_model_name,
        "prompt_hash": global_prompt_hash,
        "config_hash": global_config_hash,
    }

    async with semaphore:
        try:
            cached_payload = await asyncio.to_thread(
                visual_cache_manager.get_cached_result, entry_id, current_metadata
            )
            
            final_data_map, final_pred_sql = {}, ""

            if (cached_payload and "visual_data" in cached_payload) or eval_config.skip_visual_api_call:
                if cached_payload:
                    final_data_map = cached_payload.get("visual_data", {})
                    final_pred_sql = cached_payload.get("pred_sql", "")
                else:
                    final_data_map = ""
                    final_pred_sql = ""
            else:
                if skip_api_call:
                    logger.warning(f"Item {index}: Cache Miss & API Skipped.")
                    return False, index, PipelineStatus.P1_UNKNOWN_ERROR, ""

                rec_result, pred_sql, mm_status = await asyncio.to_thread(
                    mm_model.recognize_and_generate_sql, target_images, question
                )
                
                if rec_result:
                    final_data_map = rec_result
                    final_pred_sql = pred_sql
                    
                    await asyncio.to_thread(
                        visual_cache_manager.save_result, entry_id, 
                        {"visual_data": final_data_map, "pred_sql": final_pred_sql}, 
                        current_metadata
                    )
                else:
                    return False, index, mm_status, ""

            if not final_data_map:
                return False, index, PipelineStatus.P1_MODEL_OUTPUT_EMPTY, ""
            
            full_schema_ddl = ""
            for t_name, t_content in final_data_map.items():
                if not isinstance(t_content, dict): return False, index, PipelineStatus.P1_MODEL_OUTPUT_INVALID, ""
                _, schema, db_status = db_builder.build_db_from_data(
                    db_id=entry_id, entry_index=index, table_name=t_name, table_data=t_content
                )
                if schema: full_schema_ddl += schema + "\n"

            is_table_valid, table_entry, status = data_constructor.construct_table_entry(db_id=entry_id, visual_data=final_data_map)
            is_dev_valid, dev_entry, status = data_constructor.construct_dev_entry(new_db_id=entry_id, original_item=item)
            is_overall_success = is_table_valid and is_dev_valid
            
            if is_overall_success:
                return is_overall_success, {
                    "table_entry": table_entry,
                    "dev_entry": dev_entry,
                    "schema_ddl": full_schema_ddl
                }, PipelineStatus.SUCCESS, final_pred_sql
            else:
                return False, index, status, final_pred_sql

        except Exception as e:
            logger.error(f"Item {index} Phase 1 Error: {e}", exc_info=True)
            return False, index, PipelineStatus.P1_UNKNOWN_ERROR, ""

def initialize_e2e_run(eval_config: E2EConfig):
    e2e_prompt_template = build_e2e_prompt("{question}")
    prompt_hash = hashlib.md5(e2e_prompt_template.encode('utf-8')).hexdigest()
    
    inference_config = {
        "temp": eval_config.temperature, "top_p": eval_config.top_p,
        "freq_p": eval_config.frequency_penalty, "pres_p": eval_config.presence_penalty,
        "max_tokens": eval_config.max_tokens, "seed": eval_config.seed
    }
    config_hash = hashlib.md5(json.dumps(inference_config, sort_keys=True).encode('utf-8')).hexdigest()    
    
    run_name = f"{eval_config.e2e_model_name}_{prompt_hash[:6]}_{config_hash[:6]}"
    eval_config.setup_run_dir(run_name)
    return prompt_hash, config_hash


async def main():
    parser = argparse.ArgumentParser(description="Run E2E Evaluation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
        eval_config = E2EConfig(**yaml_data)

    prompt_hash, config_hash = initialize_e2e_run(eval_config)

    loader = EvaluationDatasetLoader(eval_config, limit=eval_config.limit)
    dataset = loader.get_data()

    mm_model = E2EMultimodalModel(eval_config)
    db_builder = DatabaseBuilder(output_dir=eval_config.generated_db_dir)
    data_constructor = PipelineDataConstructor()
    visual_cache_manager = VisualCacheManager(eval_config.visual_cache_dir)
    
    logger.info(">>> Phase 1: End-to-End Visual Extraction & SQL Generation")
    
    semaphore = asyncio.Semaphore(eval_config.e2e_concurrency)
    tasks = []
    for item in dataset:
        task = process_e2e_phase(
            semaphore, eval_config, item, mm_model, db_builder, data_constructor,
            visual_cache_manager, prompt_hash, config_hash, skip_api_call=eval_config.skip_e2e_api_call
        )
        tasks.append(task)
    
    results = await tqdm_asyncio.gather(*tasks, desc="E2E Phase")
    
    schema_memory_map = {}
    execution_status_map = {}
    failed_visual_entry = []
    pred_item_map = {}
    
    for valid, res, status, pred_sql in results:
        if valid and res:
            q_id = res['dev_entry']['question_id']
            schema_memory_map[q_id] = res['schema_ddl']
            execution_status_map[q_id] = status
            pred_item_map[str(q_id)] = pred_sql
        else:
            index = res
            failed_visual_entry.append(index)
            execution_status_map[index] = status
            pred_item_map[str(index)] = pred_sql

    schema_map_path = os.path.join(eval_config.run_dir, "visual_schemas.json")
    with open(schema_map_path, 'w', encoding='utf-8') as f: json.dump(schema_memory_map, f, ensure_ascii=False)
    
    with open(eval_config.final_prediction_path, 'w', encoding='utf-8') as f: json.dump(pred_item_map, f, indent=4, ensure_ascii=False)
    
    valid_count = len(schema_memory_map)
    logger.info(f"Phase 1 Finished. Valid Entries: {valid_count}/{len(dataset)}")
    
    logger.info(">>> Phase 3: Evaluation & Metrics Calculation")

    details_dir = os.path.join(eval_config.run_dir, "details")
    os.makedirs(details_dir, exist_ok=True)

    gold_item_map = loader.get_gt_item_map() 
    evaluator = Evaluator()
    summary_list, evaluation_results_for_log = [], []
    failed_visual_entry_set = set(failed_visual_entry)

    for index in tqdm(gold_item_map.keys(), desc="Evaluating"):
        question_id = int(index)
        gold_item = gold_item_map[index]
        
        if question_id in failed_visual_entry_set:
            status_str = str(execution_status_map.get(question_id, ""))
            
            detail_report = {
                "meta": {
                    "index": index, "question_id": question_id,
                    "question": gold_item['question'], "db_id": "",
                    "status": status_str
                },
                "visual_meta": None,
                "metrics": {"EX": 0.0, "SM": 0.0, "CM": 0.0},
                "ground_truth": {
                    "gold_schema": gold_item['gold_tables'],
                    "gold_data_dump": None,
                    "gold_sql": gold_item['SQL']
                },
                "visual_phase": {
                    "model_name": eval_config.e2e_model_name, 
                    "extracted_schema_ddl": "", 
                    "extracted_data_dump": ""
                },
                "text2sql_phase": {
                    "model_name": eval_config.e2e_model_name, 
                    "predicted_sql": "", 
                    "execution_result": "", 
                    "gold_execution_result": ""
                }
            }
            full_details_path = os.path.join(details_dir, f"{question_id}_P1_FAILED.json")
            with open(full_details_path, 'w', encoding='utf-8') as f:
                json.dump(detail_report, f, indent=4, ensure_ascii=False)

            summary_list.append({
                "index": index, "id": question_id, "status": status_str,
                "scores": {"EX": 0.0, "SM": 0.0, "CM": 0.0},
                "details_file_path": full_details_path
            })
            
            evaluation_results_for_log.append({
                "index": index, "status": {
                    "visual_success": False, "detail_status": status_str,
                    "execution_correct": False, "schema_match_score": 0.0, "content_match_score": 0.0,
                    "predicted_executed": False, "gt_executed": False, "db_built": False
                }
            })
            continue
        
        pred_sql = pred_item_map.get(str(question_id), "")
        generated_schema_str = schema_memory_map.get(question_id, "")
        pre_db_path = os.path.join(eval_config.generated_db_dir, f"{question_id}/{question_id}.sqlite")

        gt_sql = gold_item['SQL']
        gt_tables_ddl = gold_item['gold_tables']
        gt_inserts = loader.get_gold_inserts(question_id)
        
        exec_acc = False
        sm_score, cm_score = 0.0, 0.0
        pred_executed, gt_executed = False, False
        status = PipelineStatus.SUCCESS
        
        conn_pred = conn_gold = gold_db_path = None
        
        pred_data_raw = {}
        gold_data_raw = {}
        pred_res_preview = []
        gt_res_preview = []

        try:
            if not os.path.exists(pre_db_path): raise FileNotFoundError("Pred DB missing")
            
            gold_db_path = db_builder.build_gold_db(gt_tables_ddl, gt_inserts, question_id)
            if not gold_db_path: raise RuntimeError("Failed to build Gold DB")

            sm_score, best_mapping = evaluator.compare_schemas(generated_schema_str, gt_tables_ddl)
            
            pred_data_raw = db_builder.fetch_db_content(pre_db_path, None)
            gold_data_raw = db_builder.fetch_db_content(gold_db_path, set(gt_tables_ddl.keys()))
            cm_score = evaluator.compare_content(best_mapping, pred_data_raw, gold_data_raw)
            
            if pred_sql:
                try:
                    conn_pred = sqlite3.connect(f"file:{pre_db_path}?mode=ro", uri=True)
                    pred_res_preview = conn_pred.execute(pred_sql).fetchall()
                    pred_executed = True
                except: pass
            
            if gt_sql: 
                try:
                    conn_gold = sqlite3.connect(f"file:{gold_db_path}?mode=ro", uri=True)
                    gt_res_preview = conn_gold.execute(gt_sql).fetchall()
                    gt_executed = True
                except: pass

            if pred_executed and gt_executed:
                exec_acc = evaluator.compare_results(pred_res_preview, gt_res_preview)
                if not exec_acc: status = PipelineStatus.P2_RESULT_MISMATCH
            else:
                status = PipelineStatus.P2_PRED_SQL_EXEC_FAILED if not pred_executed else PipelineStatus.P2_GT_SQL_EXEC_FAILED

        except Exception as e:
            status = PipelineStatus.P2_UNKNOWN_ERROR
        finally:
            if conn_pred: conn_pred.close()
            if conn_gold: conn_gold.close()
            if gold_db_path and os.path.exists(gold_db_path):
                try: os.remove(gold_db_path)
                except: pass

        execution_status_map[index] = status
        
        detail_report = {
            "meta": {
                "index": index, "question_id": question_id,
                "question": gold_item['question'], "db_id": str(question_id),
                "status": str(status)
            },
            "visual_meta": None,
            "metrics": {
                "EX": 1.0 if exec_acc else 0.0,
                "SM": sm_score,
                "CM": cm_score
            },
            "ground_truth": {
                "gold_schema": gt_tables_ddl,
                "gold_data_dump": gold_data_raw,
                "gold_sql": gt_sql
            },
            "visual_phase": {
                "model_name": eval_config.e2e_model_name,
                "extracted_schema_ddl": generated_schema_str,
                "extracted_data_dump": pred_data_raw
            },
            "text2sql_phase": {
                "model_name": eval_config.e2e_model_name,
                "predicted_sql": pred_sql,
                "execution_result": str(pred_res_preview),
                "gold_execution_result": str(gt_res_preview)
            }
        }
        
        file_name = f"{question_id}_{question_id}.json"
        full_details_path = os.path.join(details_dir, file_name)
        with open(full_details_path, 'w', encoding='utf-8') as f:
            json.dump(detail_report, f, indent=4, ensure_ascii=False)

        summary_list.append({
            "index": index, "id": question_id, "status": str(status),
            "scores": {
                "EX": 100.0 if exec_acc else 0.0,
                "SM": round(sm_score, 2),
                "CM": round(cm_score, 2)
            },
            "details_file_path": full_details_path
        })
        
        evaluation_results_for_log.append({
            "index": index, "status": {
                "visual_success": True, "detail_status": str(status),
                "execution_correct": exec_acc, "schema_match_score": sm_score, "content_match_score": cm_score,
                "predicted_executed": pred_executed, "gt_executed": gt_executed, "db_built": os.path.exists(pre_db_path),
            }
        })

    final_metrics = evaluator.calculate_pipeline_metrics(evaluation_results_for_log)
    
    err_dist = {}
    for x in summary_list:
        s = x['status']
        err_dist[s] = err_dist.get(s, 0) + 1
    final_metrics["error_distribution_details"] = err_dist
    
    logger.info("=" * 40)
    logger.info(f"Final E2E Pipeline Summary:\n{json.dumps(final_metrics, indent=4, ensure_ascii=False)}")
    logger.info("=" * 40)
    
    with open(os.path.join(eval_config.run_dir, "final_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4, ensure_ascii=False)
        
    with open(os.path.join(eval_config.run_dir, "all_items_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())