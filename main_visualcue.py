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

# 导入包
from utils.config import PipelineConfig
from utils.constants import PipelineStatus
from utils.prompts import build_proposed_extraction_prompt
from utils.dataset_loader import EvaluationDatasetLoader
from utils.db_builder import DatabaseBuilder
from utils.pipeline_data_constructor import PipelineDataConstructor
from utils.cache_manager import VisualCacheManager, Text2SQLCacheManager
from utils.evaluator import Evaluator

from models.proposed_multimodal_model import ProposedMultimodalModel
from models.local_sota_model import LocalSOTAModel

logger = logging.getLogger('YSHLogger')


async def process_visual_phase(
    semaphore: asyncio.Semaphore,
    eval_config: PipelineConfig,
    item: Dict,
    mm_model: ProposedMultimodalModel,
    db_builder: DatabaseBuilder, 
    data_constructor: PipelineDataConstructor,
    visual_cache_manager: VisualCacheManager,
    global_prompt_hash: str,
    global_config_hash: str,
    skip_api_call: bool = True
):
    index = item.get('question_id')
    entry_id = str(index)
    question=item.get('question')

    raw_image_paths = item.get('image_paths', []) 
    if not raw_image_paths:
        logging.error(f"Item {index}: No images found.")
        return False, index, PipelineStatus.DATA_IMAGES_ERROR

    target_images = []
    for path in raw_image_paths:
        full_path = path if os.path.isabs(path) else os.path.join(eval_config.project_root, path)
        if os.path.exists(full_path):
            target_images.append(full_path)
        else:
            logger.error(f"Item {index}: Image not found: {full_path}")
            return False, index, PipelineStatus.DATA_IMAGES_ERROR

    if not target_images:
        logger.error(f"Item {index}: {target_images} is None.")
        return False, index, PipelineStatus.DATA_IMAGES_ERROR

    target_image_names = [os.path.basename(p) for p in target_images]
    target_image_names.sort()
    
    current_metadata = {
        "image_names": target_image_names,
        "level": eval_config.level,
        "model": eval_config.visual_model,
        "prompt_hash": global_prompt_hash,
        "config_hash": global_config_hash,
    }

    async with semaphore:
        try:
            cached_payload = await asyncio.to_thread(
                visual_cache_manager.get_cached_result, 
                entry_id, 
                current_metadata 
            )
            
            final_data_map = {}
            final_visual_hints = ""

            if cached_payload and isinstance(cached_payload, dict) and "tables" in cached_payload:
                final_data_map = cached_payload.get("tables", {})
                final_visual_hints = cached_payload.get("visual_hints", "")
            else:
                    
                if skip_api_call:
                    logger.warning(f"Item {index}: Cache Miss & API Skipped. Ignoring.")
                    return False, index, PipelineStatus.P1_UNKNOWN_ERROR

                rec_result, extracted_hints, mm_status = await asyncio.to_thread(
                    mm_model.recognize_table_and_hints, question, target_images
                )
                
                if rec_result:
                    final_data_map = rec_result
                    final_visual_hints = extracted_hints
                    
                    payload_to_save = {
                        "visual_hints": final_visual_hints,
                        "tables": final_data_map
                    }
                    await asyncio.to_thread(
                        visual_cache_manager.save_result, 
                        entry_id, 
                        payload_to_save, 
                        current_metadata 
                    )
                else:
                    logger.warning(f"[Item {index}] Model returned None/Empty result.")
                    return False, index, mm_status
            
            if not final_data_map:
                logger.error(f"[Item {index}] final_data_map is empty dict {{}}! Skipping.")
                return False, index, PipelineStatus.P1_MODEL_OUTPUT_INVALID
            
            full_schema_ddl = ""
            for t_name, t_content in final_data_map.items():
                if not isinstance(t_content, dict):
                    logger.error(f"[Item {index}] Table '{t_name}' content is NOT a dict.")
                    return False, index, PipelineStatus.P1_MODEL_OUTPUT_INVALID
                _, schema, db_status = db_builder.build_db_from_data(
                    db_id=entry_id,
                    entry_index=index,
                    table_name=t_name,
                    table_data=t_content,
                )
                if schema: full_schema_ddl += schema + "\n"
                else:
                    logger.warning(f"[Item {index}] Failed to build table for '{t_name}'.")
                    return False, index, db_status

            is_table_valid, table_entry, table_status = data_constructor.construct_table_entry(db_id=entry_id, visual_data=final_data_map)
            is_dev_valid, dev_entry, dev_status = data_constructor.construct_dev_entry_with_hints(new_db_id=entry_id, original_item=item, visual_hints=final_visual_hints)
            
            is_overall_success = is_table_valid and is_dev_valid
            if is_overall_success:
                return is_overall_success, {
                    "table_entry": table_entry,
                    "dev_entry": dev_entry,
                    "schema_ddl": full_schema_ddl
                }, PipelineStatus.SUCCESS
            else:
                final_error_status = table_status if not is_table_valid else dev_status
                return False, index, final_error_status

        except Exception as e:
            logger.error(f"Item {index} Phase 1 Error: {e}", exc_info=True)
            return False, index, PipelineStatus.P1_UNKNOWN_ERROR


def initialize_proposed_run(eval_config: PipelineConfig):
    logger.info("Starting PROPOSED pipeline experiment initialization...")

    visual_prompt = build_proposed_extraction_prompt(question="")
    prompt_hash = hashlib.md5(visual_prompt.encode('utf-8')).hexdigest()
    
    visual_model = eval_config.visual_model
    text2sql_model = eval_config.text2sql_model
    logger.info(f"Using visual model: {visual_model}\nUsing text2sql model: {text2sql_model}")
    
    inference_config = {
        "temp": eval_config.temperature,
        "top_p": eval_config.top_p,
        "freq_p": eval_config.frequency_penalty,
        "pres_p": eval_config.presence_penalty,
        "max_tokens": eval_config.max_tokens,
        "seed": eval_config.seed
    }

    config_str = json.dumps(inference_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()    
    
    run_name = f"proposed_{visual_model}_{prompt_hash[:6]}_{config_hash[:6]}"
    
    logger.info(f"Experiment target directory: {run_name}")
    eval_config.setup_run_dir(run_name)

    return prompt_hash, config_hash


# ==============================================================================
# main
# ==============================================================================
async def main():
    parser = argparse.ArgumentParser(description="Run Proposed Pipeline Evaluation")
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    eval_config = None
    with open(args.config, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
        if yaml_data:
            eval_config = PipelineConfig(**yaml_data)
        else:
            raise ValueError(f"{args.config}\nYAML configuration file is empty.")

    prompt_hash, config_hash = initialize_proposed_run(eval_config)

    loader = EvaluationDatasetLoader(eval_config, limit=eval_config.limit) 
    dataset = loader.get_data() 

    mm_model = ProposedMultimodalModel(eval_config) 
    db_builder = DatabaseBuilder(output_dir=eval_config.generated_db_dir) 
    data_constructor = PipelineDataConstructor()
    visual_cache_manager = VisualCacheManager(eval_config.visual_cache_dir) 
    
    logger.info("=========================================================")
    logger.info(">>> Phase 1: Visual Extraction & Layout Hints Collection")
    logger.info("=========================================================")
    
    semaphore = asyncio.Semaphore(eval_config.concurrency)
    tasks = []
    logger.info(f"Processing slice: {len(dataset)} items")
    
    for i, item in enumerate(dataset):
        task = process_visual_phase(
            semaphore, eval_config, item,
            mm_model, db_builder, data_constructor,
            visual_cache_manager, prompt_hash, config_hash,
            skip_api_call=eval_config.skip_visual_api_call
        )
        tasks.append(task)
    
    results = await tqdm_asyncio.gather(*tasks, desc="Visual Phase")
    
    generated_tables = []
    generated_dev = []
    generated_dev_sql = []

    schema_memory_map = {}
    execution_status_map = {}
    failed_visual_entry = []

    for valid, res, status in results:
        if valid and res:
            generated_tables.append(res['table_entry'])
            generated_dev.append(res['dev_entry'])
            schema_memory_map[res['dev_entry']['question_id']] = res['schema_ddl']
            execution_status_map[res['dev_entry']['question_id']] = status
            generated_dev_sql.append(res['dev_entry']['SQL'] + '\t' + res['dev_entry']['db_id'] + '\n')
        else:
            index = res
            if index is None:
                print(valid, res, status)
                exit()
            failed_visual_entry.append(index)
            execution_status_map[index] = status

    if generated_tables:
        with open(eval_config.generated_tables_json_path, 'w', encoding='utf-8') as f:
            json.dump(generated_tables, f, indent=4, ensure_ascii=False)
        with open(eval_config.generated_dev_json_path, 'w', encoding='utf-8') as f:
            json.dump(generated_dev, f, indent=4, ensure_ascii=False)
        with open(eval_config.generated_dev_sql_path, 'w', encoding='utf-8') as f:
            f.writelines(generated_dev_sql)
        logger.info(f"Intermediate files updated in: {eval_config.run_dir}")

        schema_map_path = os.path.join(eval_config.run_dir, "visual_schemas.json")
        with open(schema_map_path, 'w', encoding='utf-8') as f:
            json.dump(schema_memory_map, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved visual schemas to: {schema_map_path}")
    
    valid_count = len(generated_tables)
    failed_count = len(failed_visual_entry)
    assert valid_count + failed_count == len(dataset),\
    f'Total count mismatch: Success ({valid_count}) + Failure ({failed_count}) != Original total ({len(dataset)})'
    
    logger.info(f"Phase 1 Finished. Valid Entries: {valid_count}/{len(dataset)}, Failed Entries: {failed_count}/{len(dataset)}")
    
    if valid_count == 0:
        logger.error("Phase 1 failed to generate any valid data. Stopping Pipeline.")
        return
    
    logger.info("=========================================================")
    logger.info(">>> Phase 2: Text2SQL Inference (Batch Mode)")
    logger.info("=========================================================")
    
    input_json_path = eval_config.generated_dev_json_path
    output_json_path = eval_config.final_prediction_path
    
    if not os.path.exists(input_json_path):
        logger.error(f"Input file not found: {input_json_path}. Phase 1 failed?")
        return

    sota_name = eval_config.text2sql_model

    t2s_cache = Text2SQLCacheManager(
        input_path=eval_config.generated_tables_json_path, 
        output_path=output_json_path,
        model_name=sota_name
    )
    
    force_rerun = False
    if not force_rerun and t2s_cache.check_cache(valid_count):
        success = True 
    else:
        if force_rerun:
            logger.info("Force Rerun enabled.")
            
        t2s_model = LocalSOTAModel(eval_config)
        
        logger.info(f"Submitting batch job for model: {sota_name}")
        
        success = t2s_model.run_file_inference(
            input_path=input_json_path,
            output_path=output_json_path
        )

        if success:
            t2s_cache.save_metadata()
            logger.info(f"Pipeline Completed. Predictions saved to: {output_json_path}")
        else:
            logger.error("Phase 2 Inference Failed.")

    logger.info("=========================================================")
    logger.info(">>> Phase 3: Evaluation & Metrics Calculation (Dual-Track)")
    logger.info("=========================================================")

    if not success or not os.path.exists(output_json_path):
        logger.error("Skipping Phase 3 because Phase 2 failed or output missing.")
        return

    details_dir = os.path.join(eval_config.text2sql_run_dir, "details")
    os.makedirs(details_dir, exist_ok=True)

    schema_map = {}
    schema_map_path = os.path.join(eval_config.run_dir, "visual_schemas.json")
    if os.path.exists(schema_map_path):
        try:
            with open(schema_map_path, 'r', encoding='utf-8') as f:
                schema_map = json.load(f)
            logger.info(f"Loaded {len(schema_map)} raw schemas from visual_schemas.json")
        except Exception as e:
            logger.error(f"Failed to load schema map: {e}")
    else:
        logger.error("visual_schemas.json not found.")
        raise ValueError("'visual_schemas.json' not found.")
    
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        pred_item_map = predictions
        logger.info(f"Loaded {len(predictions)} predictions to evaluate.")
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        raise ValueError(f"Failed to load predictions: {e}")

    gold_item_map = loader.get_gt_item_map() 
    evaluator = Evaluator()
    summary_list = []              
    evaluation_results_for_log = [] 
    
    logger.info(f"Start Evaluating {len(predictions)} generated items...")
    skip_count = 0
    failed_visual_entry_set = set(failed_visual_entry)
    
    for index in tqdm(gold_item_map.keys(), desc="Evaluating"):
        logger.info(f"=-=-=-=-=-=-=-=-=-=-=-= Start evaluating data {index} =-=-=-=-=-=-=-=-=-=-=-=")
        question_id = int(index)
        db_id = index
        gold_item = gold_item_map[index] 

        if question_id in failed_visual_entry_set:
            skip_count += 1
            detail_report = {
                "meta": {
                    "index": index,
                    "question_id": question_id,
                    "question": gold_item['question'],
                    "db_id": "",
                    "status": execution_status_map.get(question_id, ""),
                },
                "visual_meta": None,
                "metrics": {"EX": 0.0, "SM": 0.0, "CM": 0.0},
                "ground_truth": {
                    "gold_schema": gold_item['gold_tables'],                
                    "gold_data_dump": None,                       
                    "gold_sql": gold_item['SQL']                            
                },
                "visual_phase": {
                    "model_name": eval_config.visual_model,                
                    "extracted_schema_ddl": "",                            
                    "extracted_data_dump": ""                              
                },
                "text2sql_phase": {
                    "model_name": eval_config.text2sql_model,              
                    "predicted_sql": "",                                   
                    "execution_result": "",                                
                    "gold_execution_result": ""
                }
            }
            
            file_name = f"{question_id}_P1_FAILED.json"
            full_details_path = os.path.join(details_dir, file_name)
            with open(full_details_path, 'w', encoding='utf-8') as f:
                json.dump(detail_report, f, indent=4, ensure_ascii=False)

            summary_list.append({
                "index": index,
                "id": question_id,
                "status": execution_status_map.get(question_id, ""),
                "scores": {"EX": 0.0, "SM": 0.0, "CM": 0.0},
                "details_file_path": full_details_path 
            })
            
            evaluation_results_for_log.append({
                "index": index,
                "status": {
                    "visual_success": False,
                    "detail_status": execution_status_map.get(question_id, ""),
                    "execution_correct": False, "schema_match_score": 0.0, "content_match_score": 0.0,
                    "predicted_executed": False, "gt_executed": False, "db_built": False,
                }
            })
            continue
        
        visual_cache_path = Path(eval_config.visual_cache_dir) / f"entry_{question_id}.json"
        with open(visual_cache_path, 'r', encoding='utf-8') as f:
            visual_cache_data = json.load(f)
            
        pred_sql = pred_item_map.get(index, "")
        generated_schema_str = schema_map.get(index, "")
        pre_db_path = os.path.join(eval_config.generated_db_dir, f"{question_id}/{question_id}.sqlite")

        gt_sql = gold_item['SQL']
        gt_tables_ddl = gold_item['gold_tables']
        gt_inserts = loader.get_gold_inserts(question_id)
        gold_db_path = None

        exec_acc, sm_score, cm_score = False, 0.0, 0.0
        status, error_msg = None, ""
        
        pred_data_raw, gold_data_raw = {}, {}
        pred_res_preview, gt_res_preview = [], []
        
        conn_pred, conn_gold = None, None
        pred_executed, gt_executed = False, False

        try:
            pred_db_exists = True
            if not os.path.exists(pre_db_path):
                pred_db_exists = False
                status = PipelineStatus.P2_PRED_DB_NOT_FOUND
                error_msg = f"Pred DB missing: {os.path.basename(pre_db_path)}"
                raise FileNotFoundError(error_msg)

            gold_db_path = db_builder.build_gold_db(gt_tables_ddl, gt_inserts, question_id)
            if not gold_db_path:
                status = PipelineStatus.P2_GT_DB_NOT_FOUND
                raise RuntimeError("Failed to build Gold DB")

            logger.info("Starting evaluation of SM metrics ....")
            sm_score, best_mapping = evaluator.compare_schemas(generated_schema_str, gt_tables_ddl)
            logger.info(f"[SM Score]: {sm_score}")

            logger.info("Starting evaluation of CM metrics.....")
            pred_data_raw = db_builder.fetch_db_content(pre_db_path, None)
            gold_data_raw = db_builder.fetch_db_content(gold_db_path, set(gt_tables_ddl.keys()))
            cm_score = evaluator.compare_content(best_mapping, pred_data_raw, gold_data_raw)
            logger.info(f"[CM Score]: {cm_score}")
            
            logger.info("Starting evaluation of EX metrics.....")
            pred_executed = False
            if pred_db_exists and pred_sql:
                try:
                    conn_pred = sqlite3.connect(f"file:{pre_db_path}?mode=ro", uri=True)
                    cursor = conn_pred.cursor()
                    cursor.execute(pred_sql)
                    pred_res_preview = cursor.fetchall() 
                    pred_executed = True
                    logger.info(f"The predicted answer is: {pred_res_preview}")
                except Exception as e:
                    status = PipelineStatus.P2_PRED_DB_NOT_FOUND
                    error_msg = f"Pred Error: {e}"
            
            gt_executed = False
            if gold_db_path and gt_sql: 
                try:
                    conn_gold = sqlite3.connect(f"file:{gold_db_path}?mode=ro", uri=True)
                    cursor = conn_gold.cursor()
                    cursor.execute(gt_sql)
                    gt_res_preview = cursor.fetchall()
                    gt_executed = True
                    logger.info(f"The label answer is: {gt_res_preview}")
                except Exception as e:
                    status = PipelineStatus.P2_GT_SQL_EXEC_FAILED
                    error_msg = f"GT Error: {e}"

            if pred_executed and gt_executed:
                exec_acc = evaluator.compare_results(pred_res_preview, gt_res_preview)
                if exec_acc:
                    status = PipelineStatus.SUCCESS
                else:
                    status = PipelineStatus.P2_RESULT_MISMATCH
                    error_msg = f"Result Mismatch: Pred Rows={pred_res_preview}, GT Rows={gt_res_preview}"
            else:
                if not pred_executed: status = PipelineStatus.P2_PRED_SQL_EXEC_FAILED
                else: status = PipelineStatus.P2_GT_SQL_EXEC_FAILED

        except Exception as e:
            if status == PipelineStatus.P2_UNKNOWN_ERROR: error_msg = str(e)
        finally:
            if conn_pred: conn_pred.close()
            if conn_gold: conn_gold.close()
            if gold_db_path and os.path.exists(gold_db_path):
                try: os.remove(gold_db_path)
                except: pass

        execution_status_map[index] = status

        detail_report = {
            "meta": {
                "index": index,
                "question_id": question_id,
                "question": gold_item['question'],
                "db_id": db_id,
                "status": str(status),
            },
            "visual_meta": visual_cache_data['meta'],
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
                "model_name": eval_config.visual_model,                
                "extracted_schema_ddl": generated_schema_str,          
                "extracted_data_dump": pred_data_raw                   
            },
            "text2sql_phase": {
                "model_name": eval_config.text2sql_model,              
                "predicted_sql": pred_sql,                             
                "execution_result": str(pred_res_preview),             
                "gold_execution_result": str(gt_res_preview)
            }
        }
        
        file_name = f"{question_id}_{db_id}.json"
        full_details_path = os.path.join(details_dir, file_name)
        
        with open(full_details_path, 'w', encoding='utf-8') as f:
            json.dump(detail_report, f, indent=4, ensure_ascii=False)

        summary_list.append({
            "index": index,
            "id": question_id,
            "status": str(status),
            "scores": {
                "EX": 100.0 if exec_acc else 0.0,
                "SM": round(sm_score, 2),
                "CM": round(cm_score, 2)
            },
            "details_file_path": full_details_path 
        })
        
        evaluation_results_for_log.append({
            "index": index,
            "status": {
                "visual_success": True,
                "detail_status": execution_status_map[index],
                "execution_correct": exec_acc,
                "schema_match_score": sm_score,
                "content_match_score": cm_score,
                "predicted_executed": pred_executed,
                "gt_executed": gt_executed,
                "db_built": os.path.exists(pre_db_path),
            }
        })

    final_metrics = evaluator.calculate_pipeline_metrics(evaluation_results_for_log)
    
    err_dist = {}
    for x in summary_list:
        s = x['status']
        err_dist[s] = err_dist.get(s, 0) + 1
    final_metrics["error_distribution_details"] = err_dist
    
    pretty_metrics = json.dumps(final_metrics, indent=4, ensure_ascii=False)
    logger.info("=" * 40)
    logger.info(f"Final Pipeline Summary:\n{pretty_metrics}")
    logger.info("=" * 40)

    with open(os.path.join(eval_config.text2sql_run_dir, "final_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4, ensure_ascii=False)
        
    with open(os.path.join(eval_config.text2sql_run_dir, "all_items_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_list, f, indent=4, ensure_ascii=False)

    logger.info(f"Evaluation Complete.")
    

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("User interrupted.")
    except Exception as e:
        logger.critical(f"Critical Error: {e}", exc_info=True)