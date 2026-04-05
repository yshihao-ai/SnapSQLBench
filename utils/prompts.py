def build_e2e_prompt(question: str) -> str:
    prompt = f"""You are an expert Data Extraction AI, Database Architect, and SQL Developer.
Your task is to detect tables in the provided image(s), extract them into a strict JSON format compatible with SQLite, and write a SQL query to answer the user's question.

**CORE TASK OVERVIEW:**
You may receive one or multiple images.
1. **Multi-Image Merging:** If multiple images appear to be vertical slices of the same long logical table, **MERGE** them into a **SINGLE** table entry.
2. **Intra-Image Table Merging vs. Separation:** If a single image contains multiple visually distinct tables, carefully evaluate if they share the same schema and logical context. If they are logically continuous parts of the same table (e.g., split by a layout break, column wrap, or formatting quirk), **MERGE** them into a single entry. Extract them as **SEPARATE** entries ONLY if they are truly independent tables with different structures or contexts.

---

## Task 1: Schema Generation & Data Extraction

For every identified table, you must extract its data and generate a standard SQLite `CREATE TABLE` statement (`schema_sql`).

### Schema & Data Rules:
* **Data Type Inference:** Analyze the cell contents to infer the best SQLite type (`INTEGER`, `REAL`, `TEXT`, `DATE`). Default to `TEXT` if uncertain.
* **Table Naming:** Use descriptive names (e.g., `report_summary`, `school_data`, or generic `table_1`).
* **Quoting:** Only use double quotes (e.g., `"col name"`) to enclose column names if they contain spaces, special characters, or are SQL keywords.
* **Headers & Occlusion:** Identify the header row. Logically infer and complete any content that is partially covered or cut off. Do not truncate data. Extract all rows visible.
* **NULL Values:** Represent logical empty cells or "None" as `null` in JSON. Do not add `NOT NULL` constraints unless explicitly stated.

---

## Task 2: SQL Query Generation

* Carefully read the user's question below.
* Based **ONLY** on the `schema_sql` you generated in Task 1, write a standard SQLite `SELECT` query to answer the question.
* Use the exact table names and column names. Use `JOIN` if multiple tables are involved.
* Ensure the query is complete and ends with a semicolon `;`.

---

**User Question:** "{question}"

**Output Structure:**
Return EXACTLY ONE ```json block. The JSON object must strictly contain two root keys:
1. `"tables"`: A dictionary containing the extracted tables, schemas (`schema_sql`), headers, and rows.
2. `"sql"`: A string containing the final SQLite query.

**Template:**
```json
{{
  "tables": {{
    "employees": {{
      "schema_sql": "CREATE TABLE employees (employee_id INTEGER, name TEXT, salary REAL, hire_date TEXT);",
      "headers": ["employee_id", "name", "salary", "hire_date"],
      "rows": [
        [101, "Alice", 60000.50, "2022-01-15"],
        [102, "Bob", 75000, "2023-05-20"]
      ]
    }},
    "departments": {{
      "schema_sql": "CREATE TABLE departments (dept_id INTEGER, dept_name TEXT);",
      "headers": ["dept_id", "dept_name"],
      "rows": [
        [1, "Sales"],
        [2, "Engineering"]
      ]
    }}
  }},
  "sql": "SELECT name FROM employees WHERE salary > 50000;"
}}
```
"""
    return prompt.strip()


def build_visual_extraction_prompt() -> str:
    prompt = """You are a specialized Data Extraction AI and Database Architect. 
Your primary task is to detect, transcribe, and define the structure of **ALL** visible tables into a strict JSON format compatible with SQLite.

**CORE TASK OVERVIEW:**
You may receive one or multiple images.
1. **Multi-Image Merging:** If multiple images appear to be vertical slices of the same long logical table, **MERGE** them into a **SINGLE** table entry.
2. **Intra-Image Table Merging vs. Separation:** If a single image contains multiple visually distinct tables, carefully evaluate if they share the same schema and logical context. If they are logically continuous parts of the same table (e.g., split by a layout break, column wrap, or formatting quirk), **MERGE** them into a single entry. Extract them as **SEPARATE** entries ONLY if they are truly independent tables with different structures or contexts.
---

## Task 1: DDL Schema Generation (CRITICAL)

For every identified table, you **MUST** generate a standard SQLite `CREATE TABLE` statement (`schema_sql`).

### Schema Rules:
* **Data Type Inference:** Analyze the cell contents to infer the best SQLite type:
    * `INTEGER`: for whole numbers, counts (if numeric).
    * `REAL`: for decimal numbers, currency, percentages.
    * `TEXT`: for names, descriptions, mixed alphanumerics.
    * `DATE`: for specific date formats (e.g., "YYYY-MM-DD", "MM/DD/YYYY", "DD-Mon-YYYY").
    * **Default:** If uncertain, use `TEXT`.
* **Table Naming:** Use descriptive names (e.g., `report_summary`, `school_data`, or generic `table_1`).
* **Quoting:** Only use double quotes (e.g., `"col name"`) to enclose column names if they contain **spaces, special characters, or are SQL keywords**. Otherwise, use unquoted names.
* **Termination:** **Every** `schema_sql` statement **MUST** end with a semicolon `;`.

---

## Task 2: Data Extraction

### Data Rules:
1.  **Headers:** Identify the header row.
2.  **Occlusion:** Logically infer and complete any content that is partially covered or cut off.

---

## Output Structure

Return a **single, raw JSON object**. 

**Template:**
```json
{
  "employees": {
    "schema_sql": "CREATE TABLE employees (employee_id INTEGER, name TEXT, salary REAL, hire_date TEXT);",
    "headers": ["employee_id", "name", "salary", "hire_date"],
    "rows": [
      [101, "Alice", 60000.50, "2022-01-15"],
      [102, "Bob", 75000, "2023-05-20"],
      [103, "Charlie", 10000, "2023-11-01"]
    ]
  },
  "departments": {
    "schema_sql": "CREATE TABLE departments (dept_id INTEGER, dept_name TEXT);",
    "headers": ["dept_id", "dept_name"],
    "rows": [
      [1, "Sales"],
      [2, "Engineering"]
    ]
  }
}
```

Action: Analyze the image(s), perform merging/separation, infer SQL types and NULL constraints, generate the DDL (schema_sql), extract the data (rows), and return the final JSON. 
""" 
    return prompt.strip()


def build_proposed_extraction_prompt(question: str) -> str:
    prompt = f"""You are an advanced Data Extraction AI and Visual Layout Analyzer.
Your primary task is to detect, transcribe, and define the structure of all visible tables. However, your task goes beyond just transcribing structure and data. The core objective is: in conjunction with the user's [Question], perform a global scan of the typography, UI elements, and visual emphasis in the image, and **"translate" these visual cues into database logic conditions that downstream Text-to-SQL models can use directly**.

**Current User Question:** "{question}"
---

**CORE TASK OVERVIEW:**
You may receive one or multiple images.
1. **Multi-Image Merging:** If multiple images appear to be vertical slices of the same long logical table, **MERGE** them into a **SINGLE** table entry.
2. **Intra-Image Table Merging vs. Separation:** If a single image contains multiple visually distinct tables, carefully evaluate if they share the same schema and logical context. If they are logically continuous parts of the same table (e.g., split by a layout break, column wrap, or formatting quirk), **MERGE** them into a single entry. Extract them as **SEPARATE** entries ONLY if they are truly independent tables with different structures or contexts.
3. **Visual Hints Extraction:** Analyze the image for specific visual information or text cues that help solve the problem (e.g., highlighted rows, handwriting, specific text information), and then "translate" these visual cues into database logic conditions directly usable by the downstream Text-to-SQL model.
---

## Task 1: DDL Schema Generation (CRITICAL)

For every identified table, you **MUST** generate a standard SQLite `CREATE TABLE` statement (`schema_sql`).

### Schema Rules:
* **Data Type Inference:** Analyze the cell contents to infer the best SQLite type:
    * `INTEGER`: for whole numbers, counts (if numeric).
    * `REAL`: for decimal numbers, currency, percentages.
    * `TEXT`: for names, descriptions, mixed alphanumerics.
    * `DATE`: for specific date formats (e.g., "YYYY-MM-DD", "MM/DD/YYYY", "DD-Mon-YYYY").
    * **Default:** If uncertain, use `TEXT`.
* **Table Naming:** Use descriptive names (e.g., `report_summary`, `school_data`, or generic `table_1`).
* **Quoting:** Only use double quotes (e.g., `"col name"`) to enclose column names if they contain **spaces, special characters, or are SQL keywords**. Otherwise, use unquoted names.
* **Termination:** **Every** `schema_sql` statement **MUST** end with a semicolon `;`.

---

## Task 2: Data Extraction

### Data Rules:
1.  **Headers:** Identify the header row.
2.  **Occlusion:** Logically infer and complete any content that is partially covered or cut off.

---

## Task 3: Derive Structured Logic Evidence (SQL-Actionable Evidence)

Closely align with the [User Question], scan the interface information, and categorize the discovered visual and typographical cues into the `visual_hints` dictionary. If no helpful cues for answering the current question are found in a specific dimension, you must enter an empty string "".

**[Step 1: Visual Scanning]** Closely align with the [User Question] and observe the image from the following three dimensions:
1. **`contextual_indicators` (Context and Perspective Indicators):**
   - Focus on global information such as bold column names, special background colors, handwritten annotations, active tabs, and window titles. Combined with the question, deduce how these cues help determine the specific business perspective of the current table or eliminate logical ambiguity in data reference within the question.
2. **`hierarchical_layout_features` (Hierarchy and Layout Features):**
   - Focus on special text styles, border lines, or indentation structures. Combined with the question, deduce whether these visual hierarchies imply rows of different data granularities. If multi-granularity data exists, deeply evaluate the calculation scope of the current question, and clarify whether an exclusion mechanism needs to be introduced (e.g., inferring filtering boundaries to exclude high-level summary records when performing basic detail aggregation).
3. **`external_annotations_and_logic` (External Annotations and Logic Definitions):**
   - Focus on supplementary text information outside the table data grid (such as mini-legends, annotation notes, or marginal notes). Combined with the question, evaluate whether this rich text information can serve as an effective supplement to the abstract table content, thereby clarifying specific filtering conditions involved in the question; or extract detailed explanations for certain business terms in the question through this text information (such as specific calculation formulas or statistical calibers).

**[Step 2: Logic Conversion]** Convert the visual information extracted above into pure logical constraints (Hints) closely related to the question, and strictly fill them into the following four dedicated fields of the `visual_hints` dictionary. If no valuable information for answering the current question is found for a specific item, you must enter an empty string "":
* `inferred_columns_and_joins`: Based on perspective indicators, determine target columns or eliminate foreign key ambiguity during multi-table joins.
* `inferred_filtering_rules`: Based on layout features and legend mapping, deduce interfering rows that need to be excluded or WHERE filtering conditions that must be added.
* `inferred_domain_logic`: Based on external annotations, extract additional business knowledge, term definitions, or specific calculation formulas required to answer the question.
* `evidence_summary`: Highly summarize all the logical constraints deduced above in one sentence as the final global hint.
---

## Output Structure

Return a **single, raw JSON object**. 

**Template:**
```json
{{
  "visual_hints": {{
    "inferred_columns_and_joins": "The 'total revenue' in the question should clearly map to the 'app_sales' column, not the 'web_sales' column; 'departure_id' should be used when joining the airport table.",
    "inferred_filtering_rules": "When performing aggregation calculations, a filtering condition must be added to exclude records where employee_name = 'Subtotal' and 'Total'; 'abnormal records' in the question correspond to the condition status_code = 3.",
    "inferred_domain_logic": "When calculating net profit, the following formula must be followed: (gross_revenue - discount) * (1 - tax_rate).",
    "evidence_summary": "When querying app_sales in this question, the Subtotal row must be excluded, only records with status_code = 3 should be filtered, and a specific formula is used to calculate the final net profit."
  }},
  "tables": {{
    "sales_records": {{
      "schema_sql": "CREATE TABLE sales_records (employee_name TEXT, departure_id INTEGER, app_sales REAL, web_sales REAL, status_code INTEGER, gross_revenue REAL, discount REAL, tax_rate REAL);",
      "headers": ["employee_name", "departure_id", "app_sales", "web_sales", "status_code", "gross_revenue", "discount", "tax_rate"],
      "rows": [
        ["Alice", 101, 5000.0, 2000.0, 3, 5000.0, 100.0, 0.05],
        ["Subtotal", null, 5000.0, 2000.0, null, 5000.0, 100.0, 0.05]
      ]
    }}
  }}
}}
```
Action: Read the user's question, comprehensively analyze the image layout and external annotations to derive SQL-actionable evidence (visual_hints), generate the DDL (schema_sql), extract the data (rows), and return the final JSON.
""" 
    return prompt.strip()