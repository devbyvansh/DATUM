import { ChangeEvent, useCallback, useEffect, useState } from "react";
import { api, AnalysisChatResponse, Chart, ChartBuildRequest, Dataset, DatasetSchema, Finding, PreparedPreview, Preview, Quality, Report, Run, SqlQueryResponse } from "./api";

type View = "workspace" | "run" | "sql";
type NoticeKind = "error" | "success" | "info";
type NoticeState = { kind: NoticeKind; text: string };
type NoticeProps = NoticeState & { onClose: () => void };

function Notice({ kind, text, onClose }: NoticeProps) {
  return (
    <div className={`notice ${kind}`}>
      <span>{text}</span>
      <button type="button" onClick={onClose} aria-label="Dismiss notice">×</button>
    </div>
  );
}

const ANALYSIS_BACKEND_STAGES = ["validate", "quality", "analysis", "verification", "final"] as const;
const PREPARATION_BACKEND_STAGES = ["validate", "quality", "preparation_plan", "approval", "transformations", "reprofile", "analysis", "verification", "final"] as const;
const ANALYSIS_TRACKER_STAGES = ["load", "process", "analyze", "visualize", "results"] as const;
const PREPARATION_TRACKER_STAGES = ["validate", "data_quality", "insights", "verify", "report"] as const;

const STAGE_LABELS: Record<string, string> = {
  load: "Load",
  process: "Process",
  analyze: "Analyze",
  visualize: "Visualize",
  results: "Results",
  validate: "Validate",
  data_quality: "Data Quality",
  insights: "Insights",
  verify: "Verify",
  report: "Report",
};

const TRACKER_COMPLETION_BACKEND_STAGE: Record<string, string> = {
  load: "validate",
  process: "quality",
  analyze: "analysis",
  visualize: "verification",
  results: "final",
  validate: "validate",
  data_quality: "quality",
  insights: "analysis",
  verify: "verification",
  report: "final",
};

const BACKEND_TO_TRACKER: Record<string, Record<string, string>> = {
  analyze_only: {
    validate: "load",
    quality: "process",
    analysis: "analyze",
    verification: "visualize",
    final: "results",
  },
  prepare_for_ml: {
    validate: "validate",
    quality: "data_quality",
    preparation_plan: "data_quality",
    approval: "verify",
    transformations: "verify",
    reprofile: "insights",
    analysis: "insights",
    verification: "verify",
    final: "report",
  },
};

function backendStageToTracker(mode: string, backendStage: string | undefined): string | null {
  return BACKEND_TO_TRACKER[mode]?.[backendStage || ""] ?? null;
}

const SUPPORTED_FILES = ".csv, .xlsx, .xls, .json, .jsonl, .parquet";

export default function App() {
  const [view, setView] = useState<View>(() => {
    const value = window.location.hash.slice(1);
    return value === "run" || value === "sql" ? value : "workspace";
  });
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [notice, setNotice] = useState<NoticeState | null>(null);
  const [busy, setBusy] = useState(false);
  const navigate = useCallback((next: View, replace = false) => {
    setView(next);
    const method = replace ? "replaceState" : "pushState";
    window.history[method]({ view: next }, "", `#${next}`);
  }, []);

  useEffect(() => {
    const handlePopState = () => {
      const value = window.location.hash.slice(1);
      setView(value === "run" || value === "sql" ? value : "workspace");
    };
    window.addEventListener("popstate", handlePopState);
    if (!window.location.hash) navigate("workspace", true);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [navigate]);

  const showError = (error: unknown) => setNotice({ kind: "error", text: error instanceof Error ? error.message : "The request failed." });

  const handleUpload = async (file: File) => {
    setBusy(true);
    setNotice(null);
    try {
      const uploaded = await api.upload(file);
      setDataset(uploaded);
      setPreview(await api.preview(uploaded.id));
      navigate("workspace");
      setNotice({ kind: "success", text: "Dataset uploaded. Choose how you want to inspect it." });
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  };

  const start = async (mode: "analyze_only" | "prepare_for_ml", preparation: Record<string, unknown> = {}) => {
    if (!dataset) return;
    setBusy(true);
    setNotice(null);
    try {
      const result = await api.startWorkflow(dataset.id, mode, preparation);
      setRun(await api.run(result.run_id));
      navigate("run");
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  };

  const openRun = async () => {
    if (!run) return;
    try {
      setRun(await api.run(run.id));
    } catch (error) {
      showError(error);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => navigate("workspace")} aria-label="Go to DATUM workspace">
          <img className="brand-logo" src="/datum-logo.png" alt="" aria-hidden="true" />
        </button>
        <span className="topbar-note">Inspect. Prepare. Analyze. Verify.</span>
        <nav className="top-nav" aria-label="Main navigation">
          <button className={view === "workspace" ? "nav-link active" : "nav-link"} onClick={() => navigate("workspace")}>Workspace</button>
          <button className={view === "sql" ? "nav-link active" : "nav-link"} onClick={() => navigate("sql")}>SQL source</button>
        </nav>
      </header>

      <main className="content">
        {notice && <Notice kind={notice.kind} text={notice.text} onClose={() => setNotice(null)} />}
        {view === "workspace" && (
          <Workspace
            dataset={dataset}
            preview={preview}
            setPreview={setPreview}
            busy={busy}
            onUpload={handleUpload}
            onStart={start}
            onOpenRun={() => navigate("workspace")}
          />
        )}
        {view === "run" && (
          <RunView
            run={run}
            dataset={dataset}
            busy={busy}
            onBack={() => navigate("workspace")}
            onRefresh={openRun}
            onDecision={async (decision) => {
              if (!run) return;
              setBusy(true);
              try {
                await api.approve(run.id, decision);
                setNotice({ kind: "success", text: decision === "approved" ? "Plan approved. DATUM is applying the selected operations." : "Plan rejected. No preparation changes were applied." });
                await new Promise((resolve) => setTimeout(resolve, 300));
                await openRun();
              } catch (error) {
                showError(error);
              } finally {
                setBusy(false);
              }
            }}
          />
        )}
        {view === "sql" && <SqlSource onBack={() => navigate("workspace")} onDataset={(next) => { setDataset(next); navigate("workspace"); }} onError={showError} />}
      </main>
    </div>
  );
}

function Workspace({ dataset, preview, setPreview, busy, onUpload, onStart, onOpenRun }: {
  dataset: Dataset | null;
  preview: Preview | null;
  setPreview: (p: Preview | null) => void;
  busy: boolean;
  onUpload: (file: File) => void;
  onStart: (mode: "analyze_only" | "prepare_for_ml", preparation?: Record<string, unknown>) => void;
  onOpenRun: () => void;
}) {
  return (
    <>
      <section className="page-heading">
        <div>
          <p className="eyebrow">WORKSPACE / START</p>
          <h1>Turn Raw Data into ML-Ready Data.</h1>
          <p className="lede">Upload a file, inspect what is in it, then choose whether DATUM should only analyze it or prepare a separate ML-ready copy.</p>
        </div>
        {dataset && <button className="button secondary" onClick={onOpenRun}>Open current run</button>}
      </section>

      {!dataset ? (
        <UploadPanel busy={busy} onUpload={onUpload} />
      ) : (
        <section className="workspace-grid">
          <div className="main-column">
            <DatasetSummary dataset={dataset} />
          <DatasetPreviewTable dataset={dataset} preview={preview} setPreview={setPreview} showControls={false} />
            <div className="choice-row">
              <button className="action-card" disabled={busy} onClick={() => onStart("analyze_only")}>
                <span className="action-kicker">NO CHANGES</span>
                <strong>Analyze dataset</strong>
                <span>Profile quality, calculate statistics, surface patterns, and verify the result without modifying the upload.</span>
                <span className="text-button">Start analysis →</span>
              </button>
              <button className="action-card accent" disabled={busy} onClick={() => onStart("prepare_for_ml")}>
                <span className="action-kicker">APPROVAL REQUIRED</span>
                <strong>Prepare data for ML</strong>
                <span>DATUM inspects this dataset, proposes suitable registered operations, and waits for your approval before writing a separate prepared file.</span>
                <span className="text-button">Generate preparation plan →</span>
              </button>
            </div>
          </div>
          <aside className="side-column">
            <div className="note-block">
              <span className="section-label">SUPPORTED INPUT</span>
              <p>{SUPPORTED_FILES}</p>
              <small>Maximum upload size follows the backend configuration. The original upload stays unchanged.</small>
            </div>
            <div className="note-block">
              <span className="section-label">NEXT STEP</span>
              <p>Review the automatically generated plan before any preparation changes run.</p>
            </div>
          </aside>
        </section>
      )}
    </>
  );
}

function UploadPanel({ busy, onUpload }: { busy: boolean; onUpload: (file: File) => void }) {
  const handle = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onUpload(file);
  };
  return (
    <section className="upload-layout">
      <label className="upload-panel">
        <input type="file" accept={SUPPORTED_FILES} onChange={handle} disabled={busy} />
        <span className="upload-symbol" aria-hidden="true">↑</span>
        <strong>{busy ? "Uploading dataset…" : "Upload a dataset"}</strong>
        <span>Drop a file here or choose one from your computer.</span>
        <small>{SUPPORTED_FILES}</small>
      </label>
      <div className="plain-list">
        <span className="section-label">WHAT HAPPENS NEXT</span>
        <div><b>01</b><span>DATUM reads the file and shows a small preview.</span></div>
        <div><b>02</b><span>You choose analysis or preparation.</span></div>
        <div><b>03</b><span>Every finding links back to a count, calculation, chart, or check.</span></div>
      </div>
    </section>
  );
}

function DatasetSummary({ dataset }: { dataset: Dataset }) {
  const meta = dataset.metadata;
  return (
    <section className="dataset-summary">
      <div className="file-line"><span className="file-icon">CSV</span><div><h2>{dataset.filename}</h2><span className="muted">{formatBytes(dataset.size_bytes)} · {dataset.metadata.source_type === "sql" ? "SQL snapshot" : "Uploaded file"}</span></div></div>
      <div className="metric-row">
        <Metric label="Rows" value={meta.row_count} />
        <Metric label="Columns" value={meta.column_count} />
        <Metric label="Missing cells" value={`${meta.missing_cell_percentage}%`} />
        <Metric label="Duplicate rows" value={meta.duplicate_row_count} />
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function DatasetPreviewTable({ dataset, preview, setPreview, showControls = true }: { dataset: Dataset; preview: Preview | null; setPreview: (p: Preview | null) => void; showControls?: boolean }) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(20);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const limit = pageSize;
  const offset = page * limit;

  const load = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const next = await api.preview(dataset.id, offset, limit, search || undefined, sortBy || undefined, sortOrder);
      setPreview(next);
      const totalPages = Math.max(1, Math.ceil(next.total_rows / next.limit));
      if (page >= totalPages) setPage(totalPages - 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dataset preview.");
    } finally {
      setBusy(false);
    }
  }, [dataset.id, offset, limit, search, sortBy, sortOrder, page, setPreview]);

  useEffect(() => { void load(); }, [load]);

  if (!preview) {
    return <div className="waiting"><span className="spinner" /> Loading dataset preview…</div>;
  }
  if (error) {
    return <div className="error-box"><b>Preview unavailable</b><p>{error}</p></div>;
  }

  const totalPages = Math.max(1, Math.ceil(preview.total_rows / preview.limit));
  const hasSearch = search.length > 0;
  const showPreviewControls = showControls !== false;

  return (
    <section className="data-section preview-table-section">
      <div className="section-heading">
        <div>
          <span className="section-label">INSPECTED DATASET</span>
          <h2>Dataset preview</h2>
          <span className="muted">{preview.total_rows} rows · {preview.columns.length} columns · showing {preview.returned_rows}</span>
        </div>
        {showPreviewControls && (
          <div className="preview-controls">
            <label className="field">
              <span>Search</span>
              <input value={search} onChange={(e) => { setSearch(e.target.value); setPage(0); }} placeholder="Search all columns…" />
            </label>
            <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(0); }}>
              <option value={20}>20 rows per page</option>
              <option value={50}>50 rows per page</option>
              <option value={100}>100 rows per page</option>
            </select>
          </div>
        )}
      </div>

      {hasSearch && <p className="muted">Showing rows matching “{search}”.</p>}

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {preview.columns.map((column) => (
                <th key={column} onClick={() => { if (sortBy === column) setSortOrder(sortOrder === "asc" ? "desc" : "asc"); else { setSortBy(column); setSortOrder("asc"); } }} style={{ cursor: "pointer" }}>
                  {column}
                  {sortBy === column && <span style={{ color: "var(--accent)", marginLeft: "6px" }}>{sortOrder === "asc" ? "↑" : "↓"}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, index) => (
              <tr key={index}>
                {preview.columns.map((column) => {
                  const value = row[column];
                  const isMissing = value === null || value === undefined || value === "";
                  return (
                    <td key={column} className={isMissing ? "missing-cell" : ""} title={isMissing ? "missing value" : String(value)}>
                      {isMissing ? <span className="missing-badge">—</span> : formatValue(value)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="preview-pagination">
        <button className="button secondary" disabled={page === 0 || busy} onClick={() => setPage(page - 1)}>← Previous</button>
        <span className="muted">Page {page + 1} of {totalPages}</span>
        <button className="button secondary" disabled={page >= totalPages - 1 || busy} onClick={() => setPage(page + 1)}>Next →</button>
        {busy && <span className="muted">loading…</span>}
      </div>
    </section>
  );
}

const CHART_TYPE_CONFIG: Record<string, {
  label: string;
  xKind?: "categorical" | "datetime" | "numeric";
  yKind?: "numeric";
  categoryKind?: "categorical";
  valueKind?: "numeric";
  metricKind?: "numeric";
  columnsKind?: "numeric";
  supportsAggregation: boolean;
  requiresX: boolean;
  requiresY: boolean;
  supportsGroupBy: boolean;
  supportsDateGranularity: boolean;
  supportsBinCount: boolean;
  supportsCorrelationMethod: boolean;
  description: string;
}> = {
  bar: {
    label: "Bar chart", xKind: "categorical", yKind: "numeric", supportsAggregation: true, requiresX: true, requiresY: true, supportsGroupBy: true, supportsDateGranularity: false, supportsBinCount: false, supportsCorrelationMethod: false,
    description: "Compare metric values across categories.",
  },
  line: {
    label: "Line chart", xKind: "datetime", yKind: "numeric", supportsAggregation: true, requiresX: true, requiresY: true, supportsGroupBy: true, supportsDateGranularity: true, supportsBinCount: false, supportsCorrelationMethod: false,
    description: "Show how a metric changes over an ordered or date axis.",
  },
  area: {
    label: "Area chart", xKind: "datetime", yKind: "numeric", supportsAggregation: true, requiresX: true, requiresY: true, supportsGroupBy: true, supportsDateGranularity: true, supportsBinCount: false, supportsCorrelationMethod: false,
    description: "Show cumulative metric values over time.",
  },
  scatter: {
    label: "Scatter plot", xKind: "numeric", yKind: "numeric", supportsAggregation: false, requiresX: true, requiresY: true, supportsGroupBy: false, supportsDateGranularity: false, supportsBinCount: false, supportsCorrelationMethod: false,
    description: "Show the relationship between two numeric fields.",
  },
  histogram: {
    label: "Histogram", xKind: "numeric", supportsAggregation: false, requiresX: true, requiresY: false, supportsGroupBy: false, supportsDateGranularity: false, supportsBinCount: true, supportsCorrelationMethod: false,
    description: "Show the distribution of a single numeric field.",
  },
  box: {
    label: "Box plot", xKind: "categorical", yKind: "numeric", supportsAggregation: false, requiresX: true, requiresY: true, supportsGroupBy: false, supportsDateGranularity: false, supportsBinCount: false, supportsCorrelationMethod: false,
    description: "Show the spread and outliers of a numeric field by category.",
  },
  pie: {
    label: "Pie chart", categoryKind: "categorical", valueKind: "numeric", supportsAggregation: true, requiresX: false, requiresY: false, supportsGroupBy: false, supportsDateGranularity: false, supportsBinCount: false, supportsCorrelationMethod: false,
    description: "Show the proportion of a metric across categories.",
  },
  correlation_heatmap: {
    label: "Correlation heatmap", columnsKind: "numeric", supportsAggregation: false, requiresX: false, requiresY: false, supportsGroupBy: false, supportsDateGranularity: false, supportsBinCount: false, supportsCorrelationMethod: true,
    description: "Show pairwise correlations between multiple numeric fields.",
  },
};

const FILTER_OPERATORS_BY_KIND: Record<string, { label: string; op: string; valueType: "single" | "multi" | "range" | "boolean" }[]> = {
  numeric: [
    { label: "equals", op: "equals", valueType: "single" },
    { label: "not equals", op: "not_equals", valueType: "single" },
    { label: "greater than", op: "greater_than", valueType: "single" },
    { label: "less than", op: "less_than", valueType: "single" },
    { label: "between", op: "between", valueType: "range" },
    { label: "is null", op: "is_empty", valueType: "boolean" },
    { label: "is not null", op: "is_not_empty", valueType: "boolean" },
  ],
  categorical: [
    { label: "equals", op: "equals", valueType: "single" },
    { label: "not equals", op: "not_equals", valueType: "single" },
    { label: "is null", op: "is_empty", valueType: "boolean" },
    { label: "is not null", op: "is_not_empty", valueType: "boolean" },
  ],
  datetime: [
    { label: "equals", op: "equals", valueType: "single" },
    { label: "not equals", op: "not_equals", valueType: "single" },
    { label: "before", op: "before", valueType: "single" },
    { label: "after", op: "after", valueType: "single" },
    { label: "between", op: "date_range", valueType: "range" },
    { label: "is null", op: "is_empty", valueType: "boolean" },
    { label: "is not null", op: "is_not_empty", valueType: "boolean" },
  ],
};

const AGGREGATION_OPTIONS = [
  { label: "None (raw values)", value: "none" },
  { label: "Sum", value: "sum" },
  { label: "Average", value: "mean" },
  { label: "Count", value: "count" },
  { label: "Minimum", value: "min" },
  { label: "Maximum", value: "max" },
];

const DATE_GRANULARITY_OPTIONS = [
  { label: "Day", value: "day" },
  { label: "Week", value: "week" },
  { label: "Month", value: "month" },
  { label: "Quarter", value: "quarter" },
  { label: "Year", value: "year" },
];

const CORRELATION_METHOD_OPTIONS = [
  { label: "Pearson", value: "pearson" },
  { label: "Spearman", value: "spearman" },
];

function columnKind(schema: DatasetSchema | null, name: string): "numeric" | "categorical" | "datetime" | "unknown" {
  if (!schema) return "unknown";
  const col = schema.schema.find((c) => c.name === name);
  if (!col) return "unknown";
  const kind = col.kind;
  if (kind === "numeric") return "numeric";
  if (kind === "datetime") return "datetime";
  return "categorical";
}

function numericColumns(schema: DatasetSchema | null): string[] {
  if (!schema) return [];
  return schema.schema.filter((c: { kind?: string; name: string }) => c.kind === "numeric").map((c: { name: string }) => c.name);
}

function categoricalColumns(schema: DatasetSchema | null): string[] {
  if (!schema) return [];
  return schema.schema.filter((c: { kind?: string; name: string }) => c.kind === "categorical").map((c: { name: string }) => c.name);
}

function datetimeColumns(schema: DatasetSchema | null): string[] {
  if (!schema) return [];
  return schema.schema.filter((c: { kind?: string; name: string }) => c.kind === "datetime").map((c: { name: string }) => c.name);
}

function ChartBuilder({ dataset, onClose }: { dataset: Dataset; onClose?: () => void }) {
  const [schema, setSchema] = useState<DatasetSchema | null>(null);
  const [schemaError, setSchemaError] = useState("");
  const [chartType, setChartType] = useState("bar");
  const [xAxis, setXAxis] = useState<string | null>(null);
  const [yAxis, setYAxis] = useState<string | null>(null);
  const [category, setCategory] = useState<string | null>(null);
  const [value, setValue] = useState<string | null>(null);
  const [metric, setMetric] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [aggregation, setAggregation] = useState("none");
  const [groupBy, setGroupBy] = useState<string | null>(null);
  const [series, setSeries] = useState<string | null>(null);
  const [filters, setFilters] = useState<Array<{ column: string; operator: string; value: any }>>([]);
  const [limit, setLimit] = useState(100);
  const [sort, setSort] = useState<string | null>(null);
  const [dateGranularity, setDateGranularity] = useState<string | null>(null);
  const [binCount, setBinCount] = useState(10);
  const [correlationMethod, setCorrelationMethod] = useState("pearson");
  const [generatedChart, setGeneratedChart] = useState<Chart | null>(null);
  const [chartBusy, setChartBusy] = useState(false);
  const [chartError, setChartError] = useState("");
  const [newFilterColumn, setNewFilterColumn] = useState<string | null>(null);
  const [newFilterOp, setNewFilterOp] = useState<string>("");
  const [newFilterValue, setNewFilterValue] = useState("");
  const [filterValueOptions, setFilterValueOptions] = useState<string[]>([]);

  const kind = columnKind(schema, xAxis || "");
  const config = CHART_TYPE_CONFIG[chartType];

  useEffect(() => {
    let active = true;
    api.datasetSchema(dataset.id)
      .then((s) => { if (active) setSchema(s); })
      .catch((err) => { if (active) setSchemaError(err instanceof Error ? err.message : "Failed to load dataset schema."); });
    return () => { active = false; };
  }, [dataset.id]);

  // Reset selections when the dataset changes to prevent stale X/Y references
  // to columns that no longer exist in the new schema.
  useEffect(() => {
    resetSelections();
  }, [dataset.id]);

  useEffect(() => {
    if (newFilterColumn && schema) {
      const k = columnKind(schema, newFilterColumn);
      setNewFilterOp(FILTER_OPERATORS_BY_KIND[k]?.[0]?.op ?? "");
      api.datasetValues(dataset.id, newFilterColumn, 100)
        .then((r) => { setFilterValueOptions(r.values); })
        .catch(() => { setFilterValueOptions([]); });
    } else {
      setFilterValueOptions([]);
    }
  }, [newFilterColumn, schema, dataset.id]);

  const numericCols = numericColumns(schema);
  const catCols = categoricalColumns(schema);
  const dateCols = datetimeColumns(schema);

  const isValidField = (col: string | null, expectedKind: "numeric" | "categorical" | "datetime"): boolean => {
    if (!col) return false;
    return columnKind(schema, col) === expectedKind;
  };

  const validationErrors: string[] = [];
  const cfg = config;
  if (cfg.requiresX && cfg.xKind && xAxis && !isValidField(xAxis, cfg.xKind)) validationErrors.push(`X axis must be a ${cfg.xKind} field.`);
  if (cfg.requiresX && cfg.xKind && !xAxis) validationErrors.push(`X axis is required for ${cfg.label}.`);
  if (cfg.requiresY && cfg.yKind && yAxis && !isValidField(yAxis, cfg.yKind)) validationErrors.push(`Y axis must be a ${cfg.yKind} field.`);
  if (cfg.requiresY && cfg.yKind && !yAxis) validationErrors.push(`Y axis is required for ${cfg.label}.`);
  if (chartType === "pie" && cfg.categoryKind && category && !isValidField(category, cfg.categoryKind)) validationErrors.push("Category must be a categorical field.");
  if (chartType === "pie" && cfg.valueKind && value && !isValidField(value, cfg.valueKind)) validationErrors.push("Value must be a numeric field.");
  if (chartType === "correlation_heatmap" && columns.length < 2) validationErrors.push("Select at least two numeric fields for correlation heatmap.");
  if (aggregation !== "none" && cfg.supportsAggregation === false) validationErrors.push(`${cfg.label} does not support aggregation.`);

  const canGenerate = validationErrors.length === 0 && schema !== null;

  const buildRequest = (): ChartBuildRequest => {
    const req: ChartBuildRequest = {
      dataset_id: dataset.id,
      chart_type: chartType,
      aggregation: aggregation === "none" ? undefined : aggregation,
      limit,
      sort: sort || undefined,
      filters: filters.length ? filters : undefined,
    };
    if (cfg.xKind && xAxis) req.x_axis = xAxis;
    if (cfg.yKind && yAxis) req.y_axis = yAxis;
    if (cfg.categoryKind && category) req.category = category;
    if (cfg.valueKind && value) req.value = value;
    if (cfg.metricKind && metric) req.metric = metric;
    if (cfg.columnsKind && columns.length) req.columns = columns;
    if (groupBy) req.group_by = groupBy;
    if (series) req.series = series;
    if (chartType === "line" || chartType === "area") req.date_granularity = dateGranularity || undefined;
    if (chartType === "histogram") req.bin_count = binCount;
    if (chartType === "correlation_heatmap") req.correlation_method = correlationMethod;
    return req;
  };

  const generateChart = async () => {
    setChartBusy(true);
    setChartError("");
    setGeneratedChart(null);
    try {
      const result = await api.buildChart(buildRequest());
      setGeneratedChart(result.chart);
    } catch (err) {
      setChartError(err instanceof Error ? err.message : "Chart generation failed.");
    } finally {
      setChartBusy(false);
    }
  };

  const addFilter = () => {
    if (!newFilterColumn || !newFilterOp) return;
    const next = [...filters, { column: newFilterColumn, operator: newFilterOp, value: newFilterValue || null }];
    setFilters(next);
    setNewFilterColumn(null);
    setNewFilterOp("");
    setNewFilterValue("");
    setFilterValueOptions([]);
  };

  const removeFilter = (index: number) => {
    setFilters(filters.filter((_, i) => i !== index));
  };

  const resetSelections = () => {
    setXAxis(null);
    setYAxis(null);
    setCategory(null);
    setValue(null);
    setMetric(null);
    setColumns([]);
    setAggregation("none");
    setGroupBy(null);
    setSeries(null);
    setSort(null);
    setDateGranularity(null);
    setBinCount(10);
    setCorrelationMethod("pearson");
    setFilters([]);
    setGeneratedChart(null);
  };

  if (schemaError) {
    return <div className="error-box"><b>Schema unavailable</b><p>{schemaError}</p></div>;
  }
  if (!schema) {
    return <div className="waiting"><span className="spinner" /> Loading dataset schema…</div>;
  }

  return (
    <section className="data-section chart-builder-section">
      <div className="section-heading">
        <div>
          <span className="section-label">BUILD A CHART</span>
          <h2>Manual Chart Builder</h2>
          <span className="muted">Create charts from your dataset using selections only — no code required.</span>
        </div>
        {onClose && <button type="button" className="button secondary" onClick={onClose}>Close builder</button>}
      </div>

      <div className="chart-builder-controls">
        <div className="cb-column">
          <div className="cb-field-group">
            <label>Chart type</label>
            <div className="cb-type-selector">
              {Object.entries(CHART_TYPE_CONFIG).map(([key, type]) => (
                <button
                  key={key}
                  className={`cb-type-btn ${chartType === key ? "cb-type-btn-active" : ""}`}
                  onClick={() => { setChartType(key); resetSelections(); }}
                >
                  {type.label}
                </button>
              ))}
            </div>
            <p className="cb-hint">{config.description}</p>
          </div>

          {config.xKind && (
            <div className="cb-field-group">
              <label>X axis <span className="cb-required">×</span></label>
              <select
                value={xAxis ?? ""}
                onChange={(e) => { setXAxis(e.target.value || null); setGeneratedChart(null); }}
                disabled={!config.xKind}
              >
                <option value="">Select a {config.xKind} field</option>
                {(config.xKind === "datetime" ? dateCols : config.xKind === "categorical" ? catCols : numericCols).map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          )}

          {config.yKind && (
            <div className="cb-field-group">
              <label>Y axis <span className="cb-required">×</span></label>
              <select
                value={yAxis ?? ""}
                onChange={(e) => { setYAxis(e.target.value || null); setGeneratedChart(null); }}
                disabled={!config.yKind}
              >
                <option value="">Select a numeric field</option>
                {numericCols.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          )}

          {chartType === "pie" && (
            <>
              <div className="cb-field-group">
                <label>Category <span className="cb-required">×</span></label>
                <select value={category ?? ""} onChange={(e) => { setCategory(e.target.value || null); setGeneratedChart(null); }}>
                  <option value="">Select a categorical field</option>
                  {catCols.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              <div className="cb-field-group">
                <label>Value (metric)</label>
                <select value={value ?? ""} onChange={(e) => { setValue(e.target.value || null); setGeneratedChart(null); }}>
                  <option value="">Count rows (default)</option>
                  {numericCols.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
            </>
          )}

          {chartType === "correlation_heatmap" && (
            <div className="cb-field-group">
              <label>Numeric fields <span className="cb-required">×</span> (2+)</label>
              <div className="cb-multi-select">
                {numericCols.map((col) => (
                  <label key={col} className="cb-pill">
                    <input
                      type="checkbox"
                      checked={columns.includes(col)}
                      onChange={(e) => {
                        if (e.target.checked) setColumns([...columns, col]);
                        else setColumns(columns.filter((c) => c !== col));
                        setGeneratedChart(null);
                      }}
                    />
                    {col}
                  </label>
                ))}
              </div>
            </div>
          )}

          {config.supportsAggregation && (
            <div className="cb-field-group">
              <label>Aggregation</label>
              <select value={aggregation} onChange={(e) => { setAggregation(e.target.value); setGeneratedChart(null); }}>
                {AGGREGATION_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              <p className="cb-hint">Optional. When disabled, raw observations are used where supported.</p>
            </div>
          )}

          {config.supportsGroupBy && numericCols.length > 0 && (
            <div className="cb-field-group">
              <label>Group by</label>
              <select value={groupBy ?? ""} onChange={(e) => { setGroupBy(e.target.value || null); setGeneratedChart(null); }}>
                <option value="">No grouping</option>
                {catCols.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          )}

          {(chartType === "line" || chartType === "area") && config.supportsDateGranularity && (
            <div className="cb-field-group">
              <label>Date granularity</label>
              <select value={dateGranularity ?? ""} onChange={(e) => { setDateGranularity(e.target.value || null); setGeneratedChart(null); }}>
                <option value="">Automatic</option>
                {DATE_GRANULARITY_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          )}

          {chartType === "histogram" && config.supportsBinCount && (
            <div className="cb-field-group">
              <label>Bins</label>
              <input type="number" min={2} max={100} value={binCount} onChange={(e) => { setBinCount(Number(e.target.value)); setGeneratedChart(null); }} />
            </div>
          )}

          {chartType === "correlation_heatmap" && config.supportsCorrelationMethod && (
            <div className="cb-field-group">
              <label>Correlation method</label>
              <select value={correlationMethod} onChange={(e) => { setCorrelationMethod(e.target.value); setGeneratedChart(null); }}>
                {CORRELATION_METHOD_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="cb-column">
          <div className="cb-field-group">
            <label>Filters</label>
            <div className="cb-filter-editor">
              <div className="cb-filter-row">
                <select
                  value={newFilterColumn ?? ""}
                  onChange={(e) => setNewFilterColumn(e.target.value || null)}
                  disabled={!schema}
                >
                  <option value="">Column</option>
                  {schema.schema.map((col) => (
                    <option key={col.name} value={col.name}>{col.name}</option>
                  ))}
                </select>
                <select value={newFilterOp} onChange={(e) => setNewFilterOp(e.target.value)} disabled={!newFilterColumn}>
                  <option value="">Operator</option>
                </select>
                {newFilterColumn && newFilterOp && (
                  <>
                    {FILTER_OPERATORS_BY_KIND[columnKind(schema, newFilterColumn)]?.find((o) => o.op === newFilterOp)?.valueType === "range" ? (
                      <input type="text" value={newFilterValue} onChange={(e) => setNewFilterValue(e.target.value)} placeholder="min,max" />
                    ) : (
                      <select
                        value={newFilterValue}
                        onChange={(e) => setNewFilterValue(e.target.value)}
                        disabled={!newFilterOp}
                      >
                        <option value="">Value</option>
                        {filterValueOptions.map((v) => (
                          <option key={v} value={v}>{v}</option>
                        ))}
                      </select>
                    )}
                    <button type="button" className="button secondary cb-btn-add" onClick={addFilter} disabled={!newFilterColumn || !newFilterOp}>Add</button>
                  </>
                )}
              </div>
              {filters.length > 0 && (
                <div className="cb-filter-list">
                  {filters.map((f, i) => (
                    <div key={i} className="cb-filter-item">
                      <span>{f.column}</span>
                      <span className="cb-filter-op">{f.operator}</span>
                      <span>{String(f.value ?? "")}</span>
                      <button type="button" className="cb-btn-remove" onClick={() => removeFilter(i)}>×</button>
                    </div>
                  ))}
                  <p className="cb-hint">All filters are combined with AND and sent to the backend for computation.</p>
                </div>
              )}
            </div>
          </div>

          <div className="cb-field-group cb-actions">
            <button className="button primary cb-btn-generate" disabled={!canGenerate || chartBusy} onClick={generateChart}>
              {chartBusy ? "Calculating…" : "Generate Chart"}
            </button>
            {validationErrors.length > 0 && (
              <div className="cb-validation-errors">
                {validationErrors.map((err, i) => (
                  <div key={i} className="cb-validation-error">⚠ {err}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {chartError && <div className="error-box"><b>Chart generation failed</b><p>{chartError}</p></div>}

      {generatedChart && (
        <div className="cb-result">
          <div className="cb-result-header">
            <h2>{generatedChart.title}</h2>
            <button className="button secondary cb-btn-close" onClick={() => setGeneratedChart(null)}>Close</button>
          </div>
          <div className="cb-result-body">
            <div className="cb-result-chart">
              <SimpleChart chart={generatedChart} />
            </div>
            <div className="cb-result-meta">
              <h3>Chart metadata</h3>
              <div className="cb-meta-grid">
                <div><span>Type</span><b>{generatedChart.type}</b></div>
                <div><span>Columns</span><b>{generatedChart.columns?.join(", ") || "—"}</b></div>
                <div><span>Aggregation</span><b>{aggregation === "none" ? "None (raw)" : aggregation}</b></div>
                <div><span>Filters</span><b>{filters.length || "None"}</b></div>
                <div><span>Dataset</span><b>{dataset.filename}</b></div>
              </div>
              {generatedChart.dataset_context && (
                <div className="cb-meta-context">
                  <h4>Dataset context</h4>
                  <DatasetContext context={generatedChart.dataset_context} />
                </div>
              )}
              {generatedChart.explanation && <p className="cb-explanation">{generatedChart.explanation}</p>}
              {generatedChart.warnings?.map((w) => (
                <p key={w} className="cb-warning">{w}</p>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function RunView({ run, dataset, busy, onBack, onRefresh, onDecision }: { run: Run | null; dataset: Dataset | null; busy: boolean; onBack: () => void; onRefresh: () => void; onDecision: (decision: "approved" | "rejected") => void }) {
  const [preparedPreview, setPreparedPreview] = useState<PreparedPreview | null>(null);
  const [preparedError, setPreparedError] = useState("");
  const [preparedPage, setPreparedPage] = useState(0);
  const [preparedColumnPage, setPreparedColumnPage] = useState(0);
  const [showChartBuilder, setShowChartBuilder] = useState(false);
  useEffect(() => {
    if (!run || ["completed", "rejected", "failed"].includes(run.status)) return;
    const timer = window.setInterval(onRefresh, 2000);
    return () => window.clearInterval(timer);
  }, [run?.id, run?.status]);
  useEffect(() => {
    if (!run || run.mode !== "prepare_for_ml" || run.status !== "completed") {
      setPreparedPreview(null);
      setPreparedError("");
      return;
    }
    let active = true;
    setPreparedError("");
    api.preparedPreview(run.id, preparedPage * 10, 10, preparedColumnPage * 8, 8)
      .then((preview) => { if (active) setPreparedPreview(preview); })
      .catch((error) => { if (active) setPreparedError(error instanceof Error ? error.message : "The prepared dataset preview is unavailable."); });
    return () => { active = false; };
  }, [run?.id, run?.mode, run?.status, preparedPage, preparedColumnPage]);

  if (!run) return <EmptyState title="No run selected" text="Start an analysis from the workspace." />;
  const report = run.result;
  return (
    <>
      <section className="page-heading run-heading">
        <div><p className="eyebrow">RUN DETAILS</p><h1>{run.mode === "analyze_only" ? "Dataset analysis" : "Preparation review"}</h1><p className="lede">{dataset?.filename ?? "Selected dataset"} · {statusCopy(run.status)}</p><small className="run-id">Run ID: {run.id}</small></div>
        <button className="button secondary" onClick={onBack}>← Back to workspace</button>
        <div className="run-actions">{!["completed", "rejected", "failed"].includes(run.status) && <button className="button secondary" onClick={onRefresh}>Refresh</button>}{report?.notebook?.available && <a className="button secondary" href={api.downloadUrl(run.id, "notebook")}>Download notebook</a>}{run.mode === "prepare_for_ml" && run.status === "completed" && <a className="button primary" href={api.downloadUrl(run.id, "prepared")}>Download prepared CSV</a>}</div>
      </section>
      <StageRail run={run} />
      {run.status === "needs_approval" && <ApprovalPanel run={run} busy={busy} onDecision={onDecision} />}
      {run.error && <div className="error-box"><b>Run failed</b><p>{run.error}</p></div>}
      {run.mode === "prepare_for_ml" && run.status === "completed" && (
        <PreparedResultsScreen preview={preparedPreview} error={preparedError} page={preparedPage} columnPage={preparedColumnPage} onPageChange={setPreparedPage} onColumnPageChange={setPreparedColumnPage} />
      )}
      {run.mode === "analyze_only" && dataset && report && run.status === "completed" && (
        <>
          {!showChartBuilder && (
            <div className="cb-toggle">
              <button className="button primary" onClick={() => setShowChartBuilder(true)}>Build a Chart</button>
            </div>
          )}
          {showChartBuilder && (
            <ChartBuilder dataset={dataset} onClose={() => setShowChartBuilder(false)} />
          )}
          <AnalyzeChartsScreen statistics={report.statistics} runId={run.id} />
        </>
      )}
      {report && run.mode !== "analyze_only" && !(run.mode === "prepare_for_ml" && run.status === "completed") && <ReportView report={report} runId={run.id} />}
      {!report && run.status !== "needs_approval" && <div className="waiting"><span className="spinner" /> DATUM is working from the source snapshot. This page checks for updates every two seconds.</div>}
    </>
  );
}

function PreparedResultsScreen({ preview, error, page, columnPage, onPageChange, onColumnPageChange }: {
  preview: PreparedPreview | null;
  error: string;
  page: number;
  columnPage: number;
  onPageChange: (page: number) => void;
  onColumnPageChange: (page: number) => void;
}) {
  return <div className="prepared-results">
    <div className="prepared-results-left"><PreparedDatasetSection preview={preview} error={error} page={page} columnPage={columnPage} onPageChange={onPageChange} onColumnPageChange={onColumnPageChange} /></div>
    <aside className="prepared-results-right">
      {preview ? <PreparedInspection preview={preview} /> : <div className="report-section"><h2>Inspection &amp; Results</h2><p className="muted">Results will appear when the prepared output is available.</p></div>}
    </aside>
  </div>;
}

function PreparedInspection({ preview }: { preview: PreparedPreview }) {
  const verificationStatus = preview.verification.status ?? "unavailable";
  const overall = verificationStatus === "failed" ? "Failed" : preview.quality_warnings.length ? "Passed with warnings" : verificationStatus === "passed" ? "Passed" : "Unavailable";
  return <div className="inspection-stack">
    <section className="inspection-section"><div className="section-heading"><div><span className="section-label">INSPECTION &amp; RESULTS</span><h2>Quality Summary</h2></div><StatusBadge value={overall.toLowerCase().replaceAll(" ", "_")} /></div>
      <div className="quality-score-inline"><b>{preview.quality_scores.prepared?.score ?? "Unavailable"}{preview.quality_scores.prepared ? "%" : ""}</b><span>Data quality score ({preview.quality_scores.original?.score ?? "?"}% → {preview.quality_scores.prepared?.score ?? "?"}%) <small title={preview.quality_scores.prepared?.limitations?.join(" ")}>Screening indicator, not ML readiness</small></span></div>
      <div className="quality-compare"><div><span>Metric</span><b>Original</b><b>Prepared</b><b>Change</b></div>{[["Rows", "rows"], ["Columns", "columns"], ["Missing cells %", "missing_cell_percentage"], ["Duplicate rows %", "duplicate_percentage"]].map(([label, key]) => { const orig = preview.comparison.original[key]; const prep = preview.comparison.prepared[key]; const change = prep - orig; return <div key={key}><span>{label}</span><b>{orig}</b><b>{prep}</b><b>{change >= 0 ? "+" : ""}{change}</b></div>; })}<div><span>Non-finite values</span><b>{preview.non_finite_values.original}</b><b>{preview.non_finite_values.prepared}</b><b>{preview.non_finite_values.prepared - preview.non_finite_values.original}</b></div></div>
      <p className="inspection-note">{preview.comparison.original.rows - preview.comparison.prepared.rows} rows removed during preparation.</p>
      {preview.quality_warnings.length > 0 && <div className="warning-list">{preview.quality_warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>}
    </section>
    <section className="inspection-section"><div className="section-heading"><div><span className="section-label">TRANSFORMATIONS</span><h2>Feature Engineering Applied</h2></div></div>{preview.transformations.length ? <div className="transformation-list">{preview.transformations.map((item, index) => <div className="transformation-item" key={`${item.operation}-${index}`}><b>{operationName(item.operation)}</b><span>{item.columns?.join(", ") || "Dataset"} · {JSON.stringify(item.parameters ?? {})}</span><small>{item.rows_before !== undefined && item.rows_after !== undefined ? `${item.rows_before - item.rows_after} rows removed` : "Executed successfully."}</small></div>)}</div> : <p className="muted">No transformations were executed.</p>}</section>
    <section className="inspection-section"><div className="section-heading"><div><span className="section-label">VERIFICATION</span><h2>Verification Results</h2></div><StatusBadge value={verificationStatus} /></div><div className="check-list">{preview.verification.checks?.map((check) => <div className="check-row" key={check.check_id}><span className={check.status === "passed" ? "check-icon pass" : "check-icon fail"}>{check.status === "passed" ? "✓" : "!"}</span><span><b>{check.check_id.replaceAll("_", " ")}</b><small>{check.status} · {JSON.stringify(check.evidence ?? {})}</small></span></div>)}</div></section>
  </div>;
}
function AnalyzeChartsScreen({ statistics, runId }: { statistics?: Report["statistics"]; runId: string }) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; text: string; result?: AnalysisChatResponse }>>([]);
  const [context, setContext] = useState<Record<string, unknown>>({});
  const [chatBusy, setChatBusy] = useState(false);
  const [chatError, setChatError] = useState("");
  const [failedQuestion, setFailedQuestion] = useState("");

  const ask = async () => {
    const text = question.trim();
    if (!text || chatBusy) return;
    setQuestion("");
    setChatError("");
    setFailedQuestion("");

    setMessages((items) => [...items, { role: "user", text }]);
    setChatBusy(true);
    try {
      const result = await api.analysisChat(runId, text, { ...context });
      setContext(result.context);
      setMessages((items) => [...items, { role: "assistant", text: result.answer, result }]);
    } catch (error) {
      setFailedQuestion(text);
      setChatError(error instanceof Error ? error.message : "The analysis request failed.");
    } finally {
      setChatBusy(false);
    }
  };

  return (
    <section className="analysis-screen">
      <div className="analysis-charts">
        <div className="section-heading">
          <div><span className="section-label">ANALYSIS</span><h2>Ask your dataset</h2></div>
        </div>
        {statistics?.warnings?.map((warning) => <p className="warning-line" key={warning}>{warning}</p>)}
        <div className="analysis-chat">
          <div className="chat-heading"><h3>Chat with your dataset</h3><span className="muted">Ask questions across all columns, records, trends, and relationships in this dataset.</span><small>Run {runId} · Full source dataset</small></div>
          <div className="chat-history" aria-live="polite">{messages.map((message, index) => <ChatMessage message={message} key={`${message.role}-${index}`} />)}</div>
          {chatError && <div className="chat-error"><p className="chart-warning">{chatError}</p><button type="button" className="button secondary" onClick={() => { setQuestion(failedQuestion); setChatError(""); }}>Retry</button></div>}
          <div className="chat-suggestions"><button type="button" onClick={() => setQuestion("What are the most important trends?")}>Key trends</button><button type="button" onClick={() => setQuestion("Which values are missing?")}>Missing values</button><button type="button" onClick={() => setQuestion("Which numerical columns are correlated?")}>Correlations</button></div>
          <form className="chat-form" onSubmit={(event) => { event.preventDefault(); void ask(); }}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask across the complete dataset" aria-label="Chat with your dataset" /><button className="button primary" disabled={chatBusy || !question.trim()}>{chatBusy ? "Calculating…" : "Ask"}</button></form>
        </div>
      </div>
    </section>
  );
}

function DatasetContext({ context }: { context: NonNullable<Chart["dataset_context"]> }) {
  return <div className="dataset-context"><span className="section-label">DATASET CONTEXT</span><div><span>Dataset</span><b>{context.dataset_name ?? "Not specified"}</b></div><div><span>Size</span><b>{context.rows ?? "?"} rows · {context.columns ?? "?"} columns</b></div><div><span>Source columns</span><b>{context.source_columns?.join(", ") || "Not specified"}</b></div><div><span>Aggregation</span><b>{context.aggregation ?? "Not specified"}</b></div><div><span>Scope</span><b>{context.record_scope ?? "Not specified"}</b></div>{context.exclusions && <small>{context.exclusions}</small>}</div>;
}

function PreparedDatasetSection({ preview, error, page, columnPage, onPageChange, onColumnPageChange }: {
  preview: PreparedPreview | null;
  error: string;
  page: number;
  columnPage: number;
  onPageChange: (page: number) => void;
  onColumnPageChange: (page: number) => void;
}) {
  if (error) return <section className="report-section prepared-dataset"><div className="section-heading"><div><span className="section-label">PREPARED OUTPUT</span><h2>Prepared Dataset</h2></div></div><div className="error-box"><b>Prepared preview unavailable</b><p>{error}</p></div></section>;
  if (!preview) return <section className="report-section prepared-dataset"><div className="section-heading"><div><span className="section-label">PREPARED OUTPUT</span><h2>Prepared Dataset</h2></div></div><div className="waiting"><span className="spinner" /> Loading the prepared dataset preview.</div></section>;
  const totalPages = Math.max(1, Math.ceil(preview.total_rows / preview.limit));
  const totalColumnPages = Math.max(1, Math.ceil(preview.total_columns / preview.column_limit));
  const metrics = [["Rows", "rows"], ["Columns", "columns"], ["Missing cells", "missing_cells"], ["Duplicate rows", "duplicate_rows"], ["Numeric columns", "numeric_columns"]] as const;
  return <section className="report-section prepared-dataset">
    <div className="section-heading">
      <div><span className="section-label">PREPARED OUTPUT</span><h2>Prepared Dataset</h2><p className="muted">Preview your data after feature engineering and cleaning.</p><span className="muted">{preview.filename} · {preview.total_rows} rows · {preview.total_columns} columns</span></div>
      <a className="button primary" href={api.downloadUrl(preview.run_id, "prepared")}>Download Prepared CSV</a>
    </div>
    <div className="comparison-grid">{metrics.map(([label, key]) => <div className="comparison-metric" key={key}><span>{label}</span><b>{preview.comparison.original[key]}</b><small>Original</small><b>{preview.comparison.prepared[key]}</b><small>Prepared</small></div>)}</div>
    {preview.new_columns.length > 0 && <div className="generated-columns"><span className="section-label">NEWLY GENERATED COLUMNS</span><p>{preview.new_columns.join(", ")}</p><small>{preview.transformations.map((item) => operationName(item.operation)).join(" · ")}</small></div>}
    <div className="section-heading prepared-table-heading"><div><h3>Prepared records</h3><span className="muted">Showing rows {preview.offset + 1}–{Math.min(preview.offset + preview.rows.length, preview.total_rows)} of {preview.total_rows} and columns {preview.column_offset + 1}–{Math.min(preview.column_offset + preview.columns.length, preview.total_columns)} of {preview.total_columns}.</span></div><div className="button-row"><button className="button secondary" disabled={columnPage === 0} onClick={() => onColumnPageChange(columnPage - 1)}>← Columns</button><button className="button secondary" disabled={columnPage >= totalColumnPages - 1} onClick={() => onColumnPageChange(columnPage + 1)}>Columns →</button></div></div>
    <div className="table-scroll prepared-table"><table><thead><tr><th>#</th>{preview.columns.map((column) => <th className={column.new ? "new-column" : ""} key={column.name}>{column.name}{column.new && <em>NEW</em>}<small>{column.dtype}</small></th>)}</tr></thead><tbody>{preview.rows.map((row, index) => <tr key={index}><td>{preview.offset + index + 1}</td>{preview.columns.map((column) => <td className={column.new ? "new-column" : ""} title={String(row[column.name] ?? "")} key={column.name}>{formatValue(row[column.name])}</td>)}</tr>)}</tbody></table></div>
    <div className="button-row prepared-pagination"><button className="button secondary" disabled={page === 0} onClick={() => onPageChange(page - 1)}>← Previous rows</button><span className="muted">Page {page + 1} of {totalPages}</span><button className="button secondary" disabled={page >= totalPages - 1} onClick={() => onPageChange(page + 1)}>Next rows →</button></div>
  </section>;
}

function StageRail({ run }: { run: Run }) {
  const stages = run.mode === "prepare_for_ml" ? PREPARATION_TRACKER_STAGES : ANALYSIS_TRACKER_STAGES;
  return (
    <ol className="stage-rail">
      {stages.map((stage) => {
        const status = getStageStatus(run, stage);
        return (
          <li className={statusClass(status)} key={stage}>
            <span className="stage-dot">{status === "completed" || status === "passed" || status === "approved" ? "✓" : status === "failed" ? "!" : "·"}</span>
            <span>{STAGE_LABELS[stage]}</span>
            <small>{stageStatusName(status)}</small>
          </li>
        );
      })}
    </ol>
  );
}

function getStageStatus(run: Run, stage: string): string {
  const trackerStages: readonly string[] = run.mode === "prepare_for_ml" ? PREPARATION_TRACKER_STAGES : ANALYSIS_TRACKER_STAGES;
  const stageIndex = trackerStages.indexOf(stage);
  if (stageIndex < 0) return "pending";

  if (run.status === "completed") return "completed";
  if (run.status === "queued") return stageIndex === 0 ? "running" : "pending";

  const backendStages: readonly string[] = run.mode === "prepare_for_ml" ? PREPARATION_BACKEND_STAGES : ANALYSIS_BACKEND_STAGES;
  const currentBackendStage = run.stage || "";
  const currentTracker = backendStageToTracker(run.mode, currentBackendStage || undefined);
  const isCurrent = currentTracker === stage;

  const completionBackendStage = TRACKER_COMPLETION_BACKEND_STAGE[stage];
  const completionIndex = completionBackendStage ? backendStages.indexOf(completionBackendStage) : -1;
  const currentIndex = currentBackendStage ? backendStages.indexOf(currentBackendStage) : -1;

  if (run.status === "failed") {
    if (isCurrent) return "failed";
    if (currentIndex >= 0 && completionIndex < currentIndex) return "completed";
    return "pending";
  }
  if (isCurrent) return "running";
  if (currentIndex >= 0 && completionIndex < currentIndex) return "completed";
  return "pending";
}

function ApprovalPanel({ run, busy, onDecision }: { run: Run; busy: boolean; onDecision: (decision: "approved" | "rejected") => void }) {
  const steps = run.result?.preparation_plan?.steps ?? [];
  return <section className="approval-panel"><div><span className="section-label">YOUR DECISION</span><h2>Review before DATUM changes a copy</h2><p>DATUM inspected the dataset and generated this plan. The original upload remains untouched. Approving writes a separate prepared CSV and records each operation.</p></div><div className="plan-list">{steps.length ? steps.map((step, index) => <div className="plan-step" key={`${step.operation}-${index}`}><span className="step-number">{String(index + 1).padStart(2, "0")}</span><div className="plan-fields"><div><b>Operation</b><span>{operationName(step.operation)}</span></div><div><b>Reason</b><span>{step.reason || "Not specified"}</span></div><div><b>Columns</b><span>{step.parameters?.column ? String(step.parameters.column) : "Dataset"}</span></div><div><b>Expected result</b><span>{step.expected_effect || "Not specified"}</span></div>{Object.keys(step.parameters ?? {}).length > 0 && <div><b>Parameters</b><span>{Object.entries(step.parameters).map(([key, value]) => `${key}: ${String(value)}`).join(" · ")}</span></div>}{step.safety_checks?.length ? <div><b>Safety note</b><span>{step.safety_checks.join(" ")}</span></div> : null}</div></div>) : <p>No transformations were proposed.</p>}</div><div className="decision-row"><button className="button danger" onClick={() => onDecision("rejected")} disabled={busy}>Reject plan</button><button className="button primary" onClick={() => onDecision("approved")} disabled={busy}>Approve and run</button></div></section>;
}

function ReportView({ report, runId }: { report: Report; runId: string }) {
  return <div className="report-layout"><QualitySection quality={report.quality} /><Insights statistics={report.statistics} runId={runId} /><VerificationSection verification={report.verification} quality={report.quality} transformations={report.transformations} aiStatus={report.ai?.status} /></div>;
}

function QualitySection({ quality }: { quality?: Quality }) {
  const findings = quality?.findings ?? [];
  const metric = (value: number | undefined, suffix = "") => value === undefined ? "Unavailable" : `${value}${suffix}`;
  return <section className="report-section"><div className="section-heading"><div><span className="section-label">DATA CHECKER</span><h2>Data quality</h2></div>{quality?.quality_score && <div className="score"><strong>{quality.quality_score.score}</strong><span>/ 100 screening score</span></div>}</div><div className="evidence-grid"><Metric label="Missing cells" value={metric(quality?.missing_cell_percentage, "%")} /><Metric label="Duplicate rows" value={metric(quality?.duplicate_percentage, "%")} /><Metric label="Infinite columns" value={quality?.infinite_columns ? quality.infinite_columns.length : "Unavailable"} /></div>{findings.length ? <div className="finding-list">{findings.map((finding, index) => <FindingRow finding={finding} key={`${finding.check_id}-${index}`} />)}</div> : <div className="empty-inline">{quality ? "No quality findings were returned for this dataset." : "Quality results are unavailable for this run."}</div>}</section>;
}

function FindingRow({ finding }: { finding: Finding }) {
  return <article className={`finding ${finding.severity ?? "info"}`}><span className="finding-badge">{finding.severity ?? "info"}</span><div><b>{finding.message}</b><p>{finding.recommendation}</p>{finding.evidence && <small>Evidence: {Object.entries(finding.evidence).map(([key, value]) => `${key} ${String(value)}`).join(" · ")}</small>}</div></article>;
}

function Insights({ statistics, runId }: { statistics?: Report["statistics"]; runId: string }) {
  const charts = statistics?.charts ?? [];
  const numeric = statistics?.descriptive_statistics ?? {};
  const [filter, setFilter] = useState("all");
  const [customCharts, setCustomCharts] = useState<Chart[] | null>(null);
  const types = [...new Set(charts.map((chart) => chart.type))];
  const visible = customCharts ?? (filter === "all" ? charts : charts.filter((chart) => chart.type === filter));
  const chooseCharts = async (value: string) => {
    setFilter(value);
    if (value === "all") {
      setCustomCharts(null);
      return;
    }
    try {
      const response = await api.customizeCharts(runId, charts.filter((chart) => chart.type === value).map((chart) => ({ id: chart.id })));
      setCustomCharts(response.charts);
    } catch {
      setCustomCharts(charts.filter((chart) => chart.type === value));
    }
  };
  return <section className="report-section"><div className="section-heading"><div><span className="section-label">INSIGHTS</span><h2>What the numbers show</h2></div>{types.length > 1 && <label className="chart-filter">Customize visualizations<select value={filter} onChange={(event) => void chooseCharts(event.target.value)}><option value="all">Recommended charts</option>{types.map((type) => <option value={type} key={type}>{type}</option>)}</select></label>}</div>{statistics?.warnings?.map((warning) => <div className="warning-line" key={warning}>{warning}</div>)}{Object.keys(numeric).length > 0 && <div className="table-scroll stats-table"><table><thead><tr><th>Column</th><th>Count</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr></thead><tbody>{Object.entries(numeric).map(([column, values]) => <tr key={column}><td><b>{column}</b></td><td>{formatValue(values.count)}</td><td>{formatValue(values.mean)}</td><td>{formatValue(values["50%"])}</td><td>{formatValue(values.min)}</td><td>{formatValue(values.max)}</td></tr>)}</tbody></table></div>}{visible.length > 0 ? <div className="chart-list">{visible.map((chart) => <SimpleChart chart={chart} key={chart.id} />)}</div> : <div className="empty-inline">{statistics ? "No compatible chart results were returned." : "Analysis results are unavailable."}</div>}</section>;
}

function SimpleChart({ chart }: { chart: Chart }) {
  const points = (chart.data ?? []).slice(0, 40);
  const values = points.map((point) => Number(point.y ?? point.value ?? 0)).filter(Number.isFinite);
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
  const imageSrc = chart.image_url ? (chart.image_url.startsWith("http") ? chart.image_url : `${API_BASE}${chart.image_url}`) : null;
  const render = () => {
    if (imageSrc) {
      return <img className="chart-image" src={imageSrc} alt={chart.title} style={{ maxWidth: "100%", height: "auto", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.05)" }} />;
    }
    if (!points.length) return <div className="empty-inline">This chart has no displayable points.</div>;
    if (chart.type === "scatter") return <svg className="plot" viewBox="0 0 640 220" role="img" aria-label={`${chart.title} scatter plot`}>{points.map((point, index) => <circle key={index} cx={24 + (Number(point.x) / Math.max(...points.map((item) => Number(item.x) || 1), 1)) * 590} cy={198 - (Number(point.y) / max) * 170} r="4" />)}</svg>;
    if (chart.type === "line") return <svg className="plot" viewBox="0 0 640 220" role="img" aria-label={`${chart.title} line chart`}><polyline points={points.map((point, index) => `${24 + index * (590 / Math.max(points.length - 1, 1))},${198 - (Number(point.y) / max) * 170}`).join(" ")} /></svg>;
    if (chart.type === "box") return <div className="box-plot"><span style={{ left: `${((Number(chart.summary?.q1) - min) / (max - min || 1)) * 100}%`, width: `${((Number(chart.summary?.q3) - Number(chart.summary?.q1)) / (max - min || 1)) * 100}%` }} /><b style={{ left: `${((Number(chart.summary?.median) - min) / (max - min || 1)) * 100}%` }} /></div>;
    if (chart.type === "heatmap") return <div className="heatmap">{points.map((point, index) => <span key={index} title={`${point.x} / ${point.y}: ${point.value}`} style={{ background: `rgba(196, 240, 95, ${Math.abs(Number(point.value) || 0)})` }} />)}</div>;
    return <div className="bars">{points.map((point, index) => <span className="bar" key={index} style={{ height: `${Math.max(4, ((Number(point.y ?? point.value) - min) / (max - min || 1)) * 100)}%` }} title={`${point.x ?? point.value ?? ""}: ${point.y ?? point.value ?? ""}`} />)}</div>;
  };
  return <article className="chart-block"><div><b>{chart.title}</b><small>{chart.type} Â· {chart.columns?.join(" against ") || chart.id}</small></div>{render()}{chart.explanation && <p className="chart-explanation">{chart.explanation}</p>}{chart.warnings?.map((warning) => <small className="chart-warning" key={warning}>{warning}</small>)}</article>;
}

function VerificationSection({ verification, quality, transformations, aiStatus }: { verification?: Report["verification"]; quality?: Quality; transformations?: Report["transformations"]; aiStatus?: string }) {
  const checks = verification?.checks ?? [];
  const passed = checks.filter((check) => check.status === "passed").length;
  const failed = checks.filter((check) => check.status === "failed").length;
  const warnings = checks.filter((check) => check.status === "warning").length;
  return <section className="report-section"><div className="section-heading"><div><span className="section-label">RESULT CHECKER</span><h2>Verification results</h2></div><StatusBadge value={verification?.status ?? "unavailable"} /></div><div className="verification-score"><b>{quality?.quality_score?.score ?? "Unavailable"}{quality?.quality_score ? "%" : ""}</b><span>Data quality <small title={quality?.quality_score?.limitations.join(" ")}>Screening indicator, not universal ML readiness</small></span></div><div className="evidence-grid"><Metric label="Checks passed" value={passed} /><Metric label="Checks failed" value={failed} /><Metric label="Checks with warnings" value={warnings} /><Metric label="Issues remaining" value={quality?.findings.filter((finding) => finding.severity === "warning" || finding.severity === "error").length ?? "Unavailable"} /></div><div className="check-list">{verification?.checks?.map((check) => <div className="check-row" key={check.check_id}><span className={check.status === "passed" ? "check-icon pass" : "check-icon fail"}>{check.status === "passed" ? "✓" : "!"}</span><span><b>{check.check_id.replaceAll("_", " ")}</b><small>{check.status} · {JSON.stringify(check.evidence ?? {})}</small></span></div>)}</div>{transformations && <div className="audit-block"><span className="section-label">TRANSFORMATION AUDIT</span><p>{transformations.length ? `${transformations.length} operation${transformations.length === 1 ? "" : "s"} were executed.` : "No transformations were executed."}</p>{transformations.map((item, index) => <div className="audit-row" key={`${item.operation}-${index}`}><b>{operationName(item.operation)}</b><span>{item.columns?.join(", ") || "dataset"} · {JSON.stringify(item.parameters ?? {})}</span></div>)}</div>}<div className="ai-note">Explanation status: <b>{aiStatus ?? "unavailable"}</b>. Calculations and verification come from deterministic backend services.</div></section>;
}

function SqlSource({ onBack, onDataset, onError }: { onBack: () => void; onDataset: (dataset: Dataset) => void; onError: (error: unknown) => void }) {
  const [url, setUrl] = useState("");
  const [schema, setSchema] = useState("");
  const [query, setQuery] = useState("SELECT * FROM your_table");
  const [limit, setLimit] = useState(1000);
  const [tables, setTables] = useState<string[]>([]);
  const [result, setResult] = useState<SqlQueryResponse | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const run = async (action: () => Promise<void>) => { setBusy(true); setMessage(""); try { await action(); } catch (error) { onError(error); } finally { setBusy(false); } };
  return <><section className="page-heading"><div><p className="eyebrow">SOURCE / SQL</p><h1>Bring a read-only query into DATUM.</h1><p className="lede">Test a supported connection, inspect tables, and materialize one bounded SELECT or WITH query as a dataset. Credentials are sent only for the request and are not saved by DATUM.</p></div><button className="button secondary" onClick={onBack}>← Back to workspace</button></section><section className="sql-layout"><div className="sql-form"><label className="field"><span>Connection URL</span><input type="password" autoComplete="off" value={url} onChange={(event) => setUrl(event.target.value)} placeholder="sqlite:///data/example.db" /><small>Keep passwords out of screenshots, logs, and browser storage.</small></label><label className="field"><span>Schema (optional)</span><input value={schema} onChange={(event) => setSchema(event.target.value)} placeholder="public" /></label><div className="button-row"><button className="button secondary" disabled={!url || busy} onClick={() => run(async () => { const response = await api.sqlTest(url, schema); setMessage(`Connected to ${response.dialect}.`); })}>Test connection</button><button className="button secondary" disabled={!url || busy} onClick={() => run(async () => { const response = await api.sqlTables(url, schema); setTables([...response.tables, ...response.views.map((view) => `${view} (view)`)]); })}>List tables</button></div>{tables.length > 0 && <div className="table-pills">{tables.map((table) => <span key={table}>{table}</span>)}</div>}<label className="field"><span>Read-only query</span><textarea rows={6} value={query} onChange={(event) => setQuery(event.target.value)} /></label><label className="field"><span>Maximum rows</span><input type="number" min={1} max={10000} value={limit} onChange={(event) => setLimit(Number(event.target.value))} /></label><div className="button-row"><button className="button secondary" disabled={!url || !query || busy} onClick={() => run(async () => setResult(await api.sqlQuery(url, query, limit)))}>Preview query</button><button className="button primary" disabled={!url || !query || busy} onClick={() => run(async () => onDataset(await api.sqlDataset(url, query, limit, "sql_source.csv")))}>Use as dataset</button></div>{message && <div className="success-line">{message}</div>}</div><div className="sql-note"><span className="section-label">BOUNDARIES</span><p>DATUM accepts one bounded read-only SELECT or WITH query. Use read-only database credentials too; this UI does not replace database authorization.</p>{result && <DatasetPreviewTable dataset={{ id: "sql", filename: "SQL query", size_bytes: 0, metadata: { row_count: result.rows.length, column_count: result.columns.length, missing_cell_percentage: 0, duplicate_row_count: 0, columns: [] } as any }} preview={{ dataset_id: "sql", columns: result.columns, rows: result.rows, total_rows: result.rows.length, returned_rows: result.rows.length, offset: 0, limit: result.rows.length }} setPreview={() => {}} />}</div></section></>;
}

function ChatMessage({ message }: { message: any }) {
  return (
    <div className={`chat-message ${message.role}`}>
      <b>{message.role === "user" ? "You" : "DATUM"}</b>
      <p>{message.text}</p>
      {message.result?.evidence?.length ? (
        <div className="chat-evidence">
          {message.result.evidence.map((item: any, idx: number) => (
            <span key={idx}>
              {item.label}: <b>{formatValue(item.value)}{item.unit ? ` ${item.unit}` : ""}</b>
            </span>
          ))}
        </div>
      ) : null}
      {message.result?.limitations?.map((item: string, idx: number) => (
        <small key={idx}>Limitation: {item}</small>
      ))}
    </div>
  );
}
function EmptyState({ title, text }: { title: string; text: string }) { return <div className="empty-state"><h2>{title}</h2><p>{text}</p></div>; }
function StatusBadge({ value }: { value: string }) { return <span className={`status-badge ${value.toLowerCase()}`}>{value.replaceAll("_", " ")}</span>; }
function statusClass(value: string) { return value === "completed" || value === "passed" || value === "approved" ? "stage completed" : value === "failed" ? "stage failed" : value === "warning" ? "stage warning" : value === "running" ? "stage running" : "stage"; }
function stageStatusName(value: string) { return value.replaceAll("_", " "); }
function statusCopy(status: string) { return status === "needs_approval" ? "waiting for your approval" : status.replaceAll("_", " "); }
function stageName(stage: string) { 
  return ({ 
    load: "Load", 
    process: "Process", 
    analyze: "Analyze", 
    visualize: "Visualize", 
    results: "Results",
    validate: "Load", 
    quality: "Process", 
    preparation_plan: "Process", 
    approval: "Analyze", 
    transformations: "Analyze", 
    reprofile: "Visualize", 
    analysis: "Visualize", 
    verification: "Results", 
    final: "Results" 
  } as Record<string, string>)[stage] ?? stage; 
}
function operationName(operation: string) { return operation.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function formatBytes(bytes: number) { return `${(bytes / 1024).toFixed(bytes < 1024 * 1024 ? 1 : 0)} ${bytes < 1024 * 1024 ? "KB" : "MB"}`; }
function formatValue(value: unknown) { if (value === null || value === undefined || value === "") return "—"; if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(3); return String(value); }
