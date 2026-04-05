import logging
import json
import re
from typing import List, Tuple, Any, Optional, Dict, Set
import time
import datetime
import sqlglot
import numpy as np
from sqlglot import parse, parse_one, exp
from dateutil import parser as date_parser
from scipy.optimize import linear_sum_assignment
logger = logging.getLogger('YSHLogger')

class Evaluator:
    def __init__(self):
        self.alpha = 0.8

    def _normalize_val(self, x: Any) -> str:
        if x is None:
            return "none"
        
        s = str(x).strip()
        s_lower = s.lower()
        
        if s_lower == "null" or s_lower == "none" or s_lower == "nan" or s == "":
            return "none"
        try:
            f_val = float(s)
            if f_val.is_integer():
                return str(int(f_val))
            rounded_val = round(f_val, 2)
            if rounded_val.is_integer():
                return str(int(rounded_val))
            return "{:.2f}".format(rounded_val).rstrip('0').rstrip('.')
        except ValueError:
            pass
        
        date_tokens = ['-', '/', '.', '年', 't', ':']
        if len(s) >= 8 and any(char.isdigit() for char in s) and any(t in s_lower for t in date_tokens):
            try:
                clean_s = s.replace('T', ' ').replace('t', ' ')
                dt = date_parser.parse(clean_s, fuzzy=False)
                
                if dt:
                    if dt.time() == datetime.time(0, 0, 0):
                        return dt.strftime("%Y-%m-%d")
                    else:
                        return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        return s_lower 

    def compare_results(self, predicted_results: Optional[List[Tuple[Any]]], 
                        ground_truth_results: Optional[List[Tuple[Any]]]) -> bool:
        """
        Compare whether the results of two SQL queries are consistent.
        """
        if predicted_results is None and ground_truth_results is None:
            return True
        
        if predicted_results is None or ground_truth_results is None:
            return False

        pred_norm = [tuple(self._normalize_val(x) for x in row) for row in predicted_results]
        gt_norm = [tuple(self._normalize_val(x) for x in row) for row in ground_truth_results]

        if pred_norm == gt_norm:
            return True

        if set(pred_norm) == set(gt_norm):
            return True
            
        return False

    def compare_schemas(self, generated_ddl_str: str, gold_schema_dict: Dict[str, str]):
        """
        Logic:
        1. Parse the DDL of Pred and Gold into structured dictionaries.
        2. Construct a Cost Matrix (score matrix) to calculate the matching score for each pair (Gold table, Pred table).
        3. Use the Hungarian Algorithm to find the globally optimal matching combination.
        4. Calculate the average score.

        Formula: SM = self.alpha * (M / |S|) + (1 - self.alpha) * (T / |S|)
        """
        if not gold_schema_dict:
            return 100.0 if not generated_ddl_str else 0.0, {}
        
        if not generated_ddl_str:
            return 0.0, {}

        gen_tables_info: Dict[str, Dict[str, str]] = {}
        try:
            parsed_expressions = parse(generated_ddl_str, dialect="sqlite")
            for expr in parsed_expressions:
                if isinstance(expr, exp.Create) and expr.kind == 'TABLE':
                    t_name = expr.this.this.name
                    cols = {}
                    for col in expr.this.expressions:
                        if isinstance(col, exp.ColumnDef):
                            c_name = col.this.name.lower()
                            c_type_raw = col.kind.this.name if col.kind else "TEXT"
                            cols[c_name] = c_type_raw
                    
                    gen_tables_info[t_name] = cols
            logger.info(f"A total of {len(gen_tables_info.keys())} predicted tables were parsed: {gen_tables_info.keys()}")
            logger.info(f"{gen_tables_info}")
        except Exception as e:
            logger.error(f"[SM] Failed to parse Generated DDL: {e}")
            return 0.0, {}

        gold_tables_info: Dict[str, Dict[str, str]] = {}
        try:
            for t_name, ddl in gold_schema_dict.items():
                expr = parse_one(ddl, dialect="sqlite")
                cols = {}
                for col in expr.this.expressions:
                    if isinstance(col, exp.ColumnDef):
                        c_name = col.this.name.lower()
                        c_type_raw = col.kind.this.name if col.kind else "TEXT"
                        cols[c_name] = c_type_raw
                gold_tables_info[t_name] = cols
        except Exception as e:
            logger.error(f"[SM] Failed to parse Gold DDL: {e}")
            raise ValueError(f"[SM] Failed to parse Gold DDL: {e}")
        
        logger.info(f"A total of {len(gold_tables_info.keys())} answer tables were parsed: {gold_tables_info.keys()}")
        logger.info(f"{gold_tables_info}")

        gold_keys = list(gold_tables_info.keys())
        gen_keys = list(gen_tables_info.keys())
        score_matrix = []

        for g_key in gold_keys:
            gold_cols = gold_tables_info[g_key]
            row_scores = []
            S_len = len(gold_cols)
            
            for gen_key in gen_keys:
                gen_cols = gen_tables_info[gen_key]
                common_cols = set(gold_cols.keys()) & set(gen_cols.keys())
                M = len(common_cols)
                if S_len == 0:
                    score = 100.0 if len(gen_cols) == 0 else 0.0
                else:
                    T = 0
                    for col_name in common_cols:
                        if gold_cols[col_name] == gen_cols[col_name]:
                            T += 1
                    
                    weighted_correct = self.alpha * M + (1 - self.alpha) * T
                    score = (weighted_correct / S_len) * 100.0
                
                row_scores.append(score)
            score_matrix.append(row_scores)
        logger.info(f"Score matrix: {score_matrix}")
        total_score = 0.0
        best_mapping = {}

        if not gen_keys:
            return 0.0, {}
        
        cost_matrix = -np.array(score_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pair = []
        for r, c in zip(row_ind, col_ind):
            total_score += score_matrix[r][c]
            gold_name = gold_keys[r]
            pre_name = gen_keys[c]
            best_mapping[gold_name] = pre_name
            pair.append((r, c))
        logger.info(f"Hungary matching result: {pair}")
        logger.info(f"schema match result: {best_mapping}")
        
        return total_score / len(gold_keys), best_mapping

    # =========================================================
    # 2: Implement Content Match Score (CM)
    # Formula: CM = K / |R|
    # K: Number of successfully matched lines (exact match)
    # |R|: Number of lines in the reference table
    # =========================================================
    def _hash_row(self, row: List[Any]) -> tuple:
        """Convert row data into hashable tuples for fast lookup"""
        return tuple(sorted(str(self._normalize_val(x)) for x in row))
    
    # =========================================================
    # Content Match Score (CM)
    # =========================================================
    def compare_content(
        self, 
        table_mapping: Dict[str, str],
        generated_data: Dict[str, List[List[Any]]], 
        gold_data_map: Dict[str, List[List[Any]]]
    ) -> float:
        """
        CM Evaluation:
        Content comparison relies on the table_mapping determined in the SM stage.  
        """
        logger.info(f"GT table name list: {list(gold_data_map.keys())}")
        logger.info(f"Pred table name list: {list(generated_data.keys())}")
        logger.info(f"Table Mapping: {table_mapping}")
        if not gold_data_map:
            return 100.0 if not generated_data else 0.0
        
        if not generated_data or not table_mapping:
            return 0.0

        total_cm_score = 0.0
        gen_data_aligned = {}
        for t_name, content in generated_data.items():
            rows = content
            row_set = set()
            for r in rows:
                if isinstance(r, (list, tuple)):
                    row_set.add(self._hash_row(r))
            gen_data_aligned[t_name] = {
                "rows_raw": rows,
                "rows_hash": row_set
            }
        
        for gt_name, gt_rows in gold_data_map.items():
            R_len = len(gt_rows)
            if R_len == 0:
                total_cm_score += 100.0
                continue
            pred_key = table_mapping.get(gt_name)
            logger.info(f"Comparing prediction table {pred_key} with answer table {gt_name}:")
            if not pred_key or pred_key not in gen_data_aligned:
                continue
            
            pred_info = gen_data_aligned[pred_key]
            gen_rows_set = pred_info["rows_hash"]
            gen_rows_raw = pred_info["rows_raw"]
            
            K = 0
            for gt_r in gt_rows:
                if self._hash_row(gt_r) in gen_rows_set:
                    K += 1
                    continue
            total_cm_score += (K / R_len) * 100.0
            logger.info(f"Score: {(K / R_len) * 100.0}")
        return total_cm_score / len(gold_data_map)


    def calculate_metrics(self, evaluation_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the final indicators, including the newly added SM and CM.
        """
        total_items = len(evaluation_outputs)
        if total_items == 0: return {"total_items": 0}

        correct_executions = 0
        
        schema_match_scores = []
        content_match_scores = []
        
        e2e_model_success = 0 
        e2e_parsing_success = 0
        db_build_success = 0
        predicted_sql_exec = 0
        gt_sql_exec = 0
        
        failed_indices = []

        for item in evaluation_outputs:
            status = item.get("status", {})
            
            if status.get("execution_correct", False): correct_executions += 1
            if "schema_match_score" in status:
                schema_match_scores.append(status["schema_match_score"])
            if "content_match_score" in status:
                content_match_scores.append(status["content_match_score"])
            if status.get("e2e_model_success"): e2e_model_success += 1
            if status.get("e2e_parsing_success"): e2e_parsing_success += 1
            if status.get("db_built"): db_build_success += 1
            if status.get("predicted_executed"): predicted_sql_exec += 1
            if status.get("gt_executed"): gt_sql_exec += 1
            
            if item.get("error_message"):
                failed_indices.append(item.get("index"))

        metrics = {
            "total_items": total_items,
            
            "Execution Accuracy (EX)": round((correct_executions / total_items) * 100, 2),
            "Schema Match (SM)": round((sum(schema_match_scores) / total_items), 2) if schema_match_scores else 0.0,
            "Content Match (CM)": round((sum(content_match_scores) / total_items), 2) if content_match_scores else 0.0,
            
            "perfect_sm_count": sum(1 for s in schema_match_scores if s > 99.9),
            "perfect_cm_count": sum(1 for s in content_match_scores if s > 99.9),
            
            "pipeline_stats": {
                "e2e_success_rate": round((e2e_model_success / total_items) * 100, 2),
                "parsing_success_rate": round((e2e_parsing_success / total_items) * 100, 2),
                "db_build_rate": round((db_build_success / total_items) * 100, 2),
                "pred_sql_exec_rate": round((predicted_sql_exec / total_items) * 100, 2),
                "gt_sql_exec_rate": round((gt_sql_exec / total_items) * 100, 2)
            }
        }
        
        logger.info(f"Final evaluation criteria:\n{json.dumps(metrics, indent=2)}")
        return metrics


    def calculate_pipeline_metrics(self, evaluation_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_items = len(evaluation_outputs)
        if total_items == 0:
            return {
                "total_items": 0,
                "error": "No evaluation data provided"
            }

        correct_executions = 0
        
        schema_match_scores = []
        content_match_scores = []
        
        cnt_db_built = 0
        cnt_pred_exec = 0
        cnt_gt_exec = 0
        cnt_visual_success = 0
        failed_indices = []

        for item in evaluation_outputs:
            status_data = item["status"]
            
            if status_data.get("execution_correct", False):
                correct_executions += 1
            
            if "schema_match_score" in status_data:
                schema_match_scores.append(status_data["schema_match_score"])
            
            if "content_match_score" in status_data:
                content_match_scores.append(status_data["content_match_score"])
                
            if status_data.get("db_built", False):
                cnt_db_built += 1
                
            if status_data.get("predicted_executed", False):
                cnt_pred_exec += 1
                
            if status_data.get("gt_executed", False):
                cnt_gt_exec += 1
            
            if status_data.get("visual_success", False):
                cnt_visual_success += 1
                
            if not status_data.get("execution_correct", False):
                idx = item["index"] 
                err_msg = status_data["detail_status"]
                failed_indices.append(f"Idx_{idx}: {err_msg}")

        def safe_div(n, d): return n / d if d > 0 else 0.0

        metrics = {
            "total_items": total_items,
            "metrics": {
                "EX (Execution Accuracy)": round(safe_div(correct_executions, total_items) * 100, 2),
                "SM (Schema Match)":       round(safe_div(sum(schema_match_scores), total_items), 2),
                "CM (Content Match)":      round(safe_div(sum(content_match_scores), total_items), 2),
            },
            "quality_control": {
                "perfect_schema_cnt": sum(1 for s in schema_match_scores if s >= 99.9),
                "perfect_content_cnt": sum(1 for c in content_match_scores if c >= 99.9),
            },

            "pipeline_funnel": {
                "visual_success_rate": f"{round(safe_div(cnt_visual_success, total_items) * 100, 1)}%",
                "db_build_rate": f"{round(safe_div(cnt_db_built, total_items) * 100, 1)}%",
                "pred_sql_validity_rate": f"{round(safe_div(cnt_pred_exec, total_items) * 100, 1)}%",
                "gt_sql_validity_rate": f"{round(safe_div(cnt_gt_exec, total_items) * 100, 1)}%"
            }
        }
        
        logger.info("="*40)
        logger.info("Pipeline Metrics Calculation Completed")
        logger.info(f"Execution Accuracy: {metrics['metrics']['EX (Execution Accuracy)']}%")
        logger.info(f"Valid SQL Generated: {cnt_pred_exec}/{total_items}")
        logger.info("="*40)
        
        return metrics