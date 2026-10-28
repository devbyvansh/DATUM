export type Status = "queued" | "running" | "needs_approval" | "completed" | "rejected" | "failed";

export interface ColumnProfile {
  name: string;
  dtype: string;
  rows: number;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  is_constant: boolean;
  potential_identifier: boolean;
  summary?: Record<string, number | null>;
  top_values?: Record<string, number>;
}

export interface Profile {
  row_count: number;
  column_count: number;
  missing_cell_count: number;
  missing_cell_percentage: number;
  duplicate_row_count: number;
  columns: ColumnProfile[];
}

export interface Dataset {
  id: string;
  filename: string;
  size_bytes: number;
  metadata: Profile & { source_type?: string };
}

export interface Finding {
  check_id?: string;
  severity?: string;
  columns?: string[];
  rows?: number;
  evidence?: Record<string, unknown>;
  message: string;
  recommendation: string;
}

export interface Quality {
  status?: string;
  findings: Finding[];
  duplicate_row_count?: number;
  duplicate_percentage?: number;
  missing_cell_count?: number;
  missing_cell_percentage?: number;
  infinite_columns?: string[];
  quality_score?: { score: number; formula: string; limitations: string[] };
}

export interface Transformation {
  operation: string;
  columns?: string[];
  parameters?: Record<string, unknown>;
  rows_before?: number;
  rows_after?: number;
}

export interface PreparationStep {
  operation: string;
  parameters: Record<string, unknown>;
  reason: string;
  requires_approval?: boolean;
  expected_effect?: string;
  safety_checks?: string[];
  expected_output_columns?: string[];
}

export interface PreparationPlan {
  status?: string;
  steps?: PreparationStep[];
}

export interface Statistics {
  descriptive_statistics?: Record<string, Record<string, number>>;
  pearson_correlations?: Record<string, Record<string, number>>;
  spearman_correlations?: Record<string, Record<string, number>>;
  categorical_frequencies?: Record<string, Record<string, number>>;
  charts?: Chart[];
  warnings?: string[];
}

export interface Chart {
  id: string;
  type: string;
  title: string;
  columns?: string[];
  data?: Array<{ value?: number; x?: string | number; y?: number | string; lower?: number; upper?: number }>;
  x_label?: string;
  y_label?: string;
  summary?: Record<string, number | null>;
  explanation?: string;
  warnings?: string[];
  image_url?: string;
  image_filename?: string;
  dataset_context?: {
    dataset_name?: string;
    rows?: number;
    columns?: number;
    source_columns?: string[];
    aggregation?: string;
    filters?: Record<string, string>;
    record_scope?: string;
    exclusions?: string;
  };
}

export interface AnalysisChatResponse {
  answer: string;
  evidence: Array<{ label: string; value: string | number; unit?: string }>;
  limitations: string[];
  chart?: Chart;
  context: {
    run_id: string;
    dataset_id: string;
    metric?: string | null;
    date_column?: string | null;
    granularity?: string;
    filters?: Record<string, string>;
    chart_id?: string | null;
  };
  provider?: { status: string; message?: string };
}

export interface Verification {
  status?: string;
  checks?: Array<{ check_id: string; status: string; evidence?: Record<string, unknown> }>;
}

export interface Report {
  run_id: string;
  dataset_id: string;
  mode: string;
  profile?: Profile;
  quality?: Quality;
  preparation_plan?: PreparationPlan;
  approval?: string;
  transformations?: Transformation[];
  statistics?: Statistics;
  verification?: Verification;
  ai?: { status: string };
  notebook?: { available: boolean; filename?: string; download?: string };
  outcome?: string;
}

export interface Run {
  id: string;
  dataset_id: string;
  mode: string;
  status: Status;
  stage: string;
  stage_status: Record<string, string>;
  approval_status?: string;
  created_at: string;
  updated_at: string;
  result?: Report;
  error?: string;
}

export interface Preview {
  dataset_id: string;
  columns: string[];
  rows: Array<Record<string, unknown>>;
  total_rows: number;
  returned_rows: number;
  offset: number;
  limit: number;
}

export interface ColumnSchema {
  name: string;
  dtype: string;
  kind?: string;
  nullable?: boolean;
  unique_count?: number;
  sample_values?: string[];
}

export interface DatasetSchema {
  dataset_id: string;
  filename: string;
  rows: number;
  columns: number;
  schema: ColumnSchema[];
}

export interface ChartBuildRequest {
  dataset_id?: string;
  chart_type: string;
  x_axis?: string;
  y_axis?: string;
  category?: string;
  value?: string;
  metric?: string;
  columns?: string[];
  aggregation?: string;
  group_by?: string;
  series?: string;
  filters?: Array<{ column: string; operator: string; value: any }>;
  limit?: number;
  sort?: string;
  date_granularity?: string;
  bin_count?: number;
  correlation_method?: string;
}

export interface SqlTableResponse {
  dialect: string;
  schema?: string | null;
  tables: string[];
  views: string[];
}

export interface SqlQueryResponse {
  dialect: string;
  row_count: number;
  columns: string[];
  rows: Array<Record<string, unknown>>;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error(
      "DATUM could not reach the backend. Start it with `python -m uvicorn backend.app.main:app --reload` and try again.",
    );
  }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail?.message ?? body.detail ?? message;
    } catch {
      // The status text is sufficient when the server does not return JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export interface PreparedPreview {
  run_id: string;
  filename: string;
  offset: number;
  limit: number;
  column_offset: number;
  column_limit: number;
  total_rows: number;
  total_columns: number;
  columns: Array<{ name: string; dtype: string; new: boolean }>;
  all_columns: string[];
  rows: Array<Record<string, unknown>>;
  new_columns: string[];
  generated_by: Record<string, string[]>;
  transformations: Transformation[];
  verification: Verification;
  quality_warnings: string[];
  quality_scores: { original?: Quality["quality_score"]; prepared?: Quality["quality_score"] };
  non_finite_values: { original: number; prepared: number };
  comparison: {
    original: Record<string, number>;
    prepared: Record<string, number>;
  };
}

export const api = {
  datasets: () => request<{ datasets: Dataset[] }>("/api/datasets"),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Dataset>("/api/datasets", { method: "POST", body });
  },
  dataset: (id: string) => request<Dataset>(`/api/datasets/${encodeURIComponent(id)}`),
  datasetSchema: (id: string) => request<DatasetSchema>(`/api/datasets/${encodeURIComponent(id)}/schema`),
  datasetValues: (id: string, column: string, limit = 50) =>
    request<{ dataset_id: string; column: string; values: string[]; total_unique: number }>(
      `/api/datasets/${encodeURIComponent(id)}/values?column=${encodeURIComponent(column)}&limit=${limit}`,
    ),
  preview: (id: string, offset = 0, limit = 20, search?: string, sort_by?: string, sort_order = "asc") =>
    request<Preview>(
      `/api/datasets/${encodeURIComponent(id)}/preview?offset=${offset}&limit=${limit}${search ? `&search=${encodeURIComponent(search)}` : ""}${sort_by ? `&sort_by=${encodeURIComponent(sort_by)}&sort_order=${sort_order}` : ""}`,
    ),
  preparedPreview: (id: string, offset = 0, limit = 10, columnOffset = 0, columnLimit = 8) =>
    request<PreparedPreview>(
      `/api/runs/${encodeURIComponent(id)}/prepared-preview?offset=${offset}&limit=${limit}&column_offset=${columnOffset}&column_limit=${columnLimit}`,
    ),
  startWorkflow: (datasetId: string, mode: string, preparation: Record<string, unknown> = {}) =>
    request<{ run_id: string }>(`/api/datasets/${encodeURIComponent(datasetId)}/workflows`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, preparation }),
    }),
  run: (id: string) => request<Run>(`/api/runs/${encodeURIComponent(id)}`),
  plan: (id: string) => request<PreparationPlan>(`/api/runs/${encodeURIComponent(id)}/preparation-plan`),
  approve: (id: string, decision: "approved" | "rejected") =>
    request<{ status: string }>(`/api/runs/${encodeURIComponent(id)}/approval`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision }),
    }),
  sqlTest: (connection_url: string, schema?: string) =>
    request<{ status: string; dialect: string }>("/api/sql/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ connection_url, schema: schema || null }),
    }),
  sqlTables: (connection_url: string, schema?: string) =>
    request<SqlTableResponse>("/api/sql/tables", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ connection_url, schema: schema || null }),
    }),
  sqlQuery: (connection_url: string, query: string, limit: number) =>
    request<SqlQueryResponse>("/api/sql/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ connection_url, query, limit }),
    }),
  sqlDataset: (connection_url: string, query: string, limit: number, filename: string) =>
    request<Dataset>("/api/sql/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ connection_url, query, limit, filename }),
    }),
  downloadUrl: (runId: string, kind: "notebook" | "prepared") =>
    `${API_BASE}/api/runs/${encodeURIComponent(runId)}/${kind === "notebook" ? "notebook" : "download"}`,
  customizeCharts: (runId: string, selections: any[]) =>
    request<{ charts: Chart[] }>(`/api/runs/${encodeURIComponent(runId)}/charts/customize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selections }),
    }),
  analysisChat: (runId: string, question: string, context: Record<string, unknown> = {}) =>
    request<AnalysisChatResponse>(`/api/runs/${encodeURIComponent(runId)}/analysis/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context }),
    }),
  buildChart: (req: ChartBuildRequest) =>
    request<{ chart: Chart; dataset: any }>("/api/charts/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
};
