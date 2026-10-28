# LangGraph workflows

`backend.app.workflows.graphs.DatumGraphs` compiles two real `StateGraph` instances with a dedicated SQLite `SqliteSaver` checkpointer and thread IDs equal to the persisted run IDs. The checkpoint database is separate from upload and prepared-data directories.

The Prepare for ML graph builds its approved operation configuration from the validated preparation plan. Unsupported operation keys and output-column collisions are rejected before a prepared artifact is written.

## Analyze only

`validate -> quality -> analysis -> verification -> final`

The graph loads the source file, profiles it, runs deterministic quality checks, calculates statistics and charts, interprets those results through the statistical agent, and runs independent verification. It has no preparation node or edge.

## Prepare for ML

`validate -> quality -> plan -> approval -> transformations -> reprofile -> analysis -> verification -> final`

The preparation agent can only produce operations in the registered transformation set accepted by `prepare_dataframe`. When a plan has steps, the approval node calls LangGraph's `interrupt`. The API stores `needs_approval` and exposes `POST /api/runs/{id}/approval`. The approval endpoint resumes the same graph thread using `Command(resume=...)`. Rejection routes to the final report without applying transformations.

SQLite checkpointing survives application reinitialization on the same machine. It remains a single-machine development design: concurrent multi-process workers and network filesystems are not supported guarantees.
