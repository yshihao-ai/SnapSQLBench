class PipelineStatus:
    """
    - P1: Phase 1 (Visual Extraction & DB Construction)
    - P2: Phase 2 (Text-to-SQL Generation & Evaluation)
    """

    SUCCESS = "SUCCESS"
    DATA_IMAGES_ERROR = "DATA_IMAGES_ERROR" 
    P1_MODEL_OUTPUT_EMPTY = "P1_MODEL_OUTPUT_EMPTY" 
    P1_MODEL_OUTPUT_INVALID = "P1_MODEL_OUTPUT_INVALID"
    P1_SCHEMA_CREATE_FAILED = "P1_SCHEMA_CREATE_FAILED"
    P1_DB_INSERTION_FAILED = "P1_DB_INSERTION_FAILED"
    P1_UNKNOWN_ERROR = "P1_UNKNOWN_ERROR"

    P2_API_ERROR = "P2_API_ERROR"
    P2_PRED_DB_NOT_FOUND = "P2_PRED_DB_NOT_FOUND"
    P2_GT_DB_NOT_FOUND = "P2_GT_DB_NOT_FOUND"
    P2_GT_SQL_EXEC_FAILED = "P2_GT_SQL_EXEC_FAILED" 
    P2_PRED_SQL_EXEC_FAILED = "P2_PRED_SQL_EXEC_FAILED"
    P2_GT_SQL_EXEC_FAILED = "P2_GT_SQL_EXEC_FAILED"

    P2_RESULT_MISMATCH = "P2_RESULT_MISMATCH" 
    P2_UNKNOWN_ERROR = "P2_UNKNOWN_ERROR"

    @classmethod
    def is_phase1_error(cls, status: str) -> bool:
        return status.startswith("P1_")

    @classmethod
    def is_phase2_error(cls, status: str) -> bool:
        return status.startswith("P2_")