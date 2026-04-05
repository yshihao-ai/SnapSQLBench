import os
from dataclasses import dataclass, field
from typing import List, Set
from .logger_init import init_logger

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)

@dataclass
class BaseConfig:
    project_root: str = field(
        default=_PROJECT_ROOT,
        metadata={"help": "Path to the project root directory"}
    )
    data_dir: str = field(
        default=os.path.join(_PROJECT_ROOT, "dataset", "BIRD", "dev"),
        metadata={"help": "Root directory of the dataset"}
        
    )
    
    visual_dev_json_path: str = field(
        default="",
        metadata={"help": "The full path of visual_dev.json."}
    )
    visual_data_json_path: str = field(
        default="",
        metadata={"help": "The full path of visual_data.json"}
    )
    gt_db_base_path: str = field(
        default="",
        metadata={"help": "Ground Truth database benchmark path"}
    )
    results_dir: str = field(
        default=os.path.join(_EVAL_DIR, "results"),
        metadata={"help": "Results Output Directory"}
    )

    use_oss: bool = field(
        default=True,
        metadata={"help": "Enable Alibaba Cloud OSS"}
    )
    oss_access_key_id: str = field(
        default="",
        metadata={"help": "OSS Access Key ID"}
    )
    oss_access_key_secret: str = field(
        default="",
        metadata={"help": "OSS Access Key Secret"}
    )
    oss_endpoint: str = field(
        default="http://oss-cn-shenzhen.aliyuncs.com",
        metadata={"help": "OSS Access Domain"}
    )
    oss_bucket_name: str = field(
        default="vt2s",
        metadata={"help": "OSS Bucket Name"}
    )
    oss_dir_prefix: str = field(
        default="eval_images/",
        metadata={"help": "Directory path prefix when uploading to OSS"}
    )

    sql_execution_timeout: int = field(
        default=15,
        metadata={"help": "SQL Execution Timeout (Seconds)"}
    )

    level: str = field(
        default="easy",
        metadata={"help": "Dataset difficulty (easy, middle, hard)."}
    )
    concurrency: str = field(
        default=8,
        metadata={"help": "Number of coroutines."}
    )
    limit: int = field(
        default=None,
        metadata={"help": "The maximum number of entries in the dataset. If it is None, there is no limit."}
    )
    skip_visual_api_call: bool = field(
        default=False,
        metadata={"help": "Whether to forcibly skip the visual recognition stage."}
    )
    def __post_init__(self):
        if not self.visual_dev_json_path:
            # /dataset/BIRD/dev/visual_dev.json
            self.visual_dev_json_path = os.path.join(self.data_dir, "visual_dev.json")
        if not self.visual_data_json_path:
            # /dataset/BIRD/dev/visual_data.json
            self.visual_data_json_path = os.path.join(self.data_dir, "visual_data.json")
        if not self.gt_db_base_path:
            # /dataset/BIRD/dev/dev_databases/
            self.gt_db_base_path = os.path.join(self.data_dir, "dev_databases")
        os.makedirs(self.results_dir, exist_ok=True)


@dataclass
class PipelineConfig(BaseConfig):

    visual_model: str = field(
        default="qwen3-vl-plus",
        metadata={"help": "Visual Model Name"}
    )
    visual_model_base_url: str = field(
        default="",
        metadata={"help": "URL of the visual model API"}
    )
    visual_model_api_key: str = field(
        default="",
        metadata={"help": "Visual Model API Key"}
    )
    
    text2sql_model_base_url: str = field(
        default="",
        metadata={"help": "The URL of the text2sql model API; if empty, the visual_model_base_url will be used"}
    )
    text2sql_model_api_key: str = field(
        default="",
        metadata={"help": "text2sql visual model API key, if empty, visual_model_api_key will be used"}
    )

    max_width: str = field(
        default=2048,
        metadata={"help": "Maximum width of image input"}
    )
    max_height: str = field(
        default=4096,
        metadata={"help": "Maximum height of image input"}
    )
    overlap: str = field(
        default=200,
        metadata={"help": "The overlap between image slices"}
    )
    
    text2sql_model: str = field(
        default="DAIL-SQL",
        metadata={"help": "Name of the model or method used in the Text2SQL stage"}
    )
    
    temperature: float = field(
        default=0.0, 
        metadata={"help": "Sampling temperature, 0 means greedy decoding"}
    )
    top_p: float = field(
        default=0.1, 
        metadata={"help": "Top-p Threshold"}
    )
    frequency_penalty: float = field(
        default=0.0, 
        metadata={"help": "Frequency penalty, reduce the probability of repeated words"}
    )
    presence_penalty: float = field(
        default=0.0, 
        metadata={"help": "There is punishment, encouraging the model to discuss new topics"}
    )
    max_tokens: int = field(
        default=16384, 
        metadata={"help": "Maximum number of tokens generated at one time"}
    )
    seed: int = field(
        default=42, 
        metadata={"help": "seed"}
    )
    
    def __post_init__(self):
        super().__post_init__()

    def setup_run_dir(self, run_name):
        pipeline_root = os.path.join(self.results_dir, "pipeline_" + self.level)
        self.run_dir = os.path.join(pipeline_root, run_name)
        self.text2sql_run_dir = os.path.join(self.run_dir, self.text2sql_model)
        
        self.visual_cache_dir = os.path.join(self.run_dir, "visual_cache")
        self.generated_db_dir = os.path.join(self.run_dir, "generated_databases")
        self.pipeline_log_file = os.path.join(self.text2sql_run_dir, f"{self.text2sql_model}_pipeline.log")

        self.generated_tables_json_path = os.path.join(self.run_dir, "generated_dev_tables.json")
        self.generated_dev_json_path = os.path.join(self.run_dir, "generated_dev.json")
        self.generated_dev_sql_path = os.path.join(self.run_dir, "generated_dev.sql")

        self.final_prediction_path = os.path.join(self.text2sql_run_dir, "final_predictions.json")
        self.execution_status_path = os.path.join(self.text2sql_run_dir, "execution_status.json")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.text2sql_run_dir, exist_ok=True)
        os.makedirs(self.visual_cache_dir, exist_ok=True)
        os.makedirs(self.generated_db_dir, exist_ok=True)

        logger = init_logger(self.pipeline_log_file)
        logger.info(f"Pipeline Run Directory: {self.run_dir}")


@dataclass
class E2EConfig(BaseConfig):
    e2e_api_endpoint: str = field(
        default="",
        metadata={"help": "E2E model API node"}
    )
    e2e_api_key: str = field(
        default="",
        metadata={"help": "E2E Model API Key"}
    )
    e2e_model_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the end-to-end model used"}
    )
    e2e_concurrency: int = field(
        default=10,
        metadata={"help": "API concurrent request count"}
    )

    temperature: float = field(
        default=0.0, 
        metadata={"help": "Model sampling temperature"}
    )
    
    top_p: float = field(
        default=0.1, 
        metadata={"help": "Top-p Threshold"}
    )

    frequency_penalty: float = field(
        default=0.0, 
        metadata={"help": "frequency penalty"}
    )
    
    presence_penalty: float = field(
        default=0.0, 
        metadata={"help": "presence penalty"}
    )
    
    max_tokens: int = field(
        default=16384, 
        metadata={"help": "Maximum tokens generated at one time"}
    )
    
    seed: int = field(
        default=42, 
        metadata={"help": "seed"}
    )
    
    max_width: int = field(
        default=2048, 
        metadata={"help": "Maximum image input width"}
    )

    max_height: int = field(
        default=4096, 
        metadata={"help": "Maximum image input height"}
    )
    
    overlap: int = field(
        default=200, 
        metadata={"help": "The overlap between image slices"}
    )
    
    skip_e2e_api_call: bool = field(
        default=False, 
        metadata={"help": "Whether to skip API call"}
    )

    def setup_run_dir(self, run_name):
        e2e_root = os.path.join(self.results_dir, "e2e_" + self.level)
        self.run_dir = os.path.join(e2e_root, run_name)
        
        self.visual_cache_dir = os.path.join(self.run_dir, "e2e_cache")
        self.generated_db_dir = os.path.join(self.run_dir, "generated_databases")
        self.pipeline_log_file = os.path.join(self.run_dir, f"{self.e2e_model_name}_e2e.log")

        self.generated_tables_json_path = os.path.join(self.run_dir, "generated_dev_tables.json")
        self.generated_dev_json_path = os.path.join(self.run_dir, "generated_dev.json")
        self.generated_dev_sql_path = os.path.join(self.run_dir, "generated_dev.sql")

        self.final_prediction_path = os.path.join(self.run_dir, "final_predictions.json")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.visual_cache_dir, exist_ok=True)
        os.makedirs(self.generated_db_dir, exist_ok=True)

        logger = init_logger(self.pipeline_log_file)
        logger.info(f"E2E Run Directory: {self.run_dir}")

