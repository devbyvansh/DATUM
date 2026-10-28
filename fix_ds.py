with open(r'D:\DATUM\frontend\src\App.tsx') as f:
    s = f.read()

old = """function DatasetSummary({ dataset }: { dataset: Dataset }) {
  const meta = dataset.metadata;
  return (
    <section className=\"dataset-summary\">
      <div className=\"file-line\"><span className=\"file-name\">{dataset.filename}</span><span className=\"file-meta\">{formatBytes(dataset.size_bytes)}{dataset.metadata.source_type ?  · source:  : \"\"}</span></div>
      <div className=\"metric-row\">
        <Metric label=\"Rows\" value={meta.row_count} />
        <Metric label=\"Columns\" value={meta.column_count} />
        <Metric label=\"Missing cells\" value={${meta.missing_cell_percentage}%} />
        <Metric label=\"Duplicate rows\" value={meta.duplicate_row_count} />
      </div>
    </section>
  );
}"""

new = """function DatasetSummary({ dataset }: { dataset: Dataset }) {
  const meta = dataset.metadata as any;
  const findings = meta.findings || [];
  return (
    <section className=\"dataset-summary\">
      <div className=\"section-heading\">
        <div>
          <span className=\"section-label\">INSPECTED DATASET</span>
          <h2>{dataset.filename}</h2>
          <p className=\"muted\">{formatBytes(dataset.size_bytes)}{dataset.metadata.source_type ?  · source:  : \"\"}</p>
        </div>
      </div>
      <div className=\"metric-row\">
        <Metric label=\"Rows\" value={meta.row_count} />
        <Metric label=\"Columns\" value={meta.column_count} />
        <Metric label=\"Missing cells\" value={${meta.missing_cell_percentage}%} />
        <Metric label=\"Duplicate rows\" value={meta.duplicate_row_count} />
      </div>
      {meta.columns && meta.columns.length > 0 && (
        <div className=\"table-scroll\" style={{ marginTop: \"1rem\" }}>
          <table>
            <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead>
            <tbody>
              {meta.columns.slice(0, 50).map((col: any) => (
                <tr key={col.name}><td><b>{col.name}</b></td><td>{col.dtype}</td><td>{col.missing_count} ({col.missing_percentage}%)</td><td>{col.unique_count}</td></tr>
              ))}
            </tbody>
          </table>
          {meta.columns.length > 50 && <p className=\"muted\">Showing first 50 of {meta.columns.length} columns</p>}
        </div>
      )}
      {findings.length > 0 && (
        <div style={{ marginTop: \"1rem\" }}>
          <span className=\"section-label\">IMPORTANT QUALITY FINDINGS</span>
          <div className=\"finding-list\">
            {findings.slice(0, 30).map((finding: any, idx: number) => (
              <FindingRow finding={finding} key={${finding.check_id}-} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}"""

if old not in s:
    print('not found')
else:
    s2 = s.replace(old, new)
    open(r'D:\DATUM\frontend\src\App.tsx','w').write(s2)
    print('ok')
