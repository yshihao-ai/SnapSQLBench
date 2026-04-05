import os
import requests
import logging
from typing import Dict, Any, Optional
logger = logging.getLogger('YSHLogger')

class LocalSOTAModel:
    def __init__(self, eval_config):
        self.model_name = eval_config.text2sql_model
        self.eval_config = eval_config
        self.service_configs = {
            'GEN-SQL':   {"host": "http://localhost:8087", "endpoint": "/GEN-SQL/run_batch_file"},
            'MAC-SQL':   {"host": "http://localhost:8089", "endpoint": "/MAC-SQL/run_batch_file"},
            'CodeS': {"host": "http://localhost:8090", "endpoint": "/CodeS/run_batch_file"},

        }
        
        if self.model_name not in self.service_configs:
            raise ValueError(f"Model '{self.model_name}' is not configured.")
            
        self.config = self.service_configs[self.model_name]
        self.api_url = f"{self.config['host']}{self.config['endpoint']}"

    def run_file_inference(self, input_path: str, output_path: str) -> bool:
        """
        Dispatch the entire task to a specific strategy method based on the model name.
        """
        abs_input = os.path.abspath(input_path)
        abs_output = os.path.abspath(output_path)
        abs_db_root = os.path.abspath(self.eval_config.generated_db_dir)

        logger.info(f"Starting inference for model: {self.model_name}")

        if self.model_name == 'MAC-SQL':
            return self._run_strategy_mac_sql()
        elif self.model_name == 'GEN-SQL':
            return self._run_strategy_gen_sql()
        elif self.model_name == 'CodeS':
            return self._run_strategy_codes_sql()
        else:
            raise NotImplementedError(f"Strategy for {self.model_name} is not implemented.")

    # =========================================================================
    # Strategy 1: MAC-SQL full-process handling
    # =========================================================================
    def _run_strategy_mac_sql(self) -> bool:
        """
        MAC-SQL Custom Logic (New Backend Adaptation):
        Calls the /MAC-SQL/run endpoint provided by mac_sql_service.py.
        Parameters are retrieved entirely from eval_config, with the payload 
        mapping to the MacSqlRequest model on the server side.
        """
        payload = {
            "gen_dev": os.path.abspath(self.eval_config.generated_dev_json_path),
            "gen_tables": os.path.abspath(self.eval_config.generated_tables_json_path),
            "gen_dbs": os.path.abspath(self.eval_config.generated_db_dir),             
            "gen_dev_sql": os.path.abspath(self.eval_config.generated_dev_sql_path),
            
            "api_key": self.eval_config.text2sql_model_api_key if self.eval_config.text2sql_model_api_key != "" else self.eval_config.visual_model_api_key,          
            "api_base": self.eval_config.text2sql_model_base_url if self.eval_config.text2sql_model_base_url != "" else self.eval_config.visual_model_base_url,
            # "api_base": "",
            
            "output": os.path.abspath(self.eval_config.final_prediction_path)
        }

        logger.info(f"[MAC-SQL] Sending request to service. Output target: {payload['output']}")

        response = self._send_post_request(self.api_url, payload)

        if response and response.status_code == 200:
            try:
                resp_json = response.json()
                if resp_json.get("status") == "success":
                    logger.info(f"[MAC-SQL] Pipeline finished successfully. File saved at: {resp_json.get('output_file')}")
                    return True
                else:
                    logger.error(f"[MAC-SQL] Service Error: {resp_json}")
                    return False
            except Exception as e:
                logger.error(f"[MAC-SQL] Invalid JSON response: {e}")
                return False
        
        err_msg = response.text if response else "No Response"
        status_code = response.status_code if response else "None"
        logger.error(f"[MAC-SQL] Request failed with status: {status_code}, Msg: {err_msg}")
        return False

    # =========================================================================
    # Strategy 2: Gen-SQL full-process handling
    # =========================================================================
    def _run_strategy_gen_sql(self) -> bool:
        """
        Gen-SQL Strategy Invocation Logic:
        Calls the /GEN-SQL/run_batch_file endpoint provided by gensql_service.py.
        Enforces the conversion of all paths to absolute paths to ensure the 
        server-side file discovery.
        """
        payload = {
            "gen_dev": os.path.abspath(self.eval_config.generated_dev_json_path),
            "gen_tables": os.path.abspath(self.eval_config.generated_tables_json_path),
            "gen_dbs": os.path.abspath(self.eval_config.generated_db_dir),            
            "gen_dev_sql": os.path.abspath(self.eval_config.generated_dev_sql_path),
            "api_key": self.eval_config.text2sql_model_api_key if self.eval_config.text2sql_model_api_key != "" else self.eval_config.visual_model_api_key,          
            "api_base": self.eval_config.text2sql_model_base_url if self.eval_config.text2sql_model_base_url != "" else self.eval_config.visual_model_base_url,
            "model": "Qwen2.5-Coder-32B-Instruct",
            "output": os.path.abspath(self.eval_config.final_prediction_path)
        }

        logger.info(f"[Gen-SQL] Sending request to service. Model: {payload['model']}")
        logger.info(f"[Gen-SQL] Input Path (Abs): {payload['gen_dev']}")

        response = self._send_post_request(self.api_url, payload)

        if response and response.status_code == 200:
            try:
                resp_json = response.json()
                if resp_json.get("status") == "success":
                    logger.info(f"[Gen-SQL] Pipeline finished successfully. Output: {payload['output']}")
                    return True
                else:
                    logger.error(f"[Gen-SQL] Service Error: {resp_json}")
                    return False
            except Exception as e:
                logger.error(f"[Gen-SQL] Invalid JSON response: {e}")
                return False
        
        logger.error(f"[Gen-SQL] Request failed with status: {response.status_code if response else 'None'}")
        return False


    # =========================================================================
    # Strategy 3: CodeS full-process handling
    # =========================================================================
    def _run_strategy_codes_sql(self) -> bool:
        """
        CodeS Custom Logic:
        Calls the /CodeS/run_pipeline endpoint provided by codes_service.py.
        Parameters are retrieved entirely from eval_config, with the payload 
        mapping to the PipelineRequest model.
        """
        using_evidence = True
        payload = {
            "gen_dev": os.path.abspath(self.eval_config.generated_dev_json_path),
            "gen_tables": os.path.abspath(self.eval_config.generated_tables_json_path),
            "gen_dbs": os.path.abspath(self.eval_config.generated_db_dir),             
            "gen_dev_sql": os.path.abspath(self.eval_config.generated_dev_sql_path),
            "output": os.path.abspath(self.eval_config.final_prediction_path),
            "evidence": True
        }

        logger.info(f"[CodeS] Sending request to service.")
        if using_evidence:
            logger.info(f"[CodeS] Using evidence.")
        else:
            logger.info(f"[CodeS] Without evidence.")
        response = self._send_post_request(self.api_url, payload)

        if response and response.status_code == 200:
            try:
                resp_json = response.json()
                if resp_json.get("status") == "success":
                    logger.info(f"[CodeS] Pipeline finished successfully. Output: {os.path.abspath(self.eval_config.final_prediction_path)}")
                    return True
                else:
                    logger.error(f"[CodeS] Service Error: {resp_json}")
                    return False
            except Exception as e:
                logger.error(f"[CodeS] Invalid JSON response: {e}")
                return False
        logger.error(f"[CodeS] Request failed with status: {response.status_code if response else 'None'}")
        return False
    
    
    def _send_post_request(self, url: str, json_data: Dict, headers: Optional[Dict] = None) -> Optional[requests.Response]:
        try:
            logger.info(f"POST -> {url}")
            resp = requests.post(url, json=json_data, headers=headers, timeout=36000000)
            return resp
        except requests.exceptions.ConnectionError:
            logger.critical(f"Connection Failed to {url}")
        except requests.exceptions.ReadTimeout:
            logger.error("Request Timed Out (Server processing took too long)")
        except Exception as e:
            logger.error(f"Network Exception: {e}")
        return None