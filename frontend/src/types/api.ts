export interface ModuleStatus { module?: string; status: string; model?: string }

export interface IngestionStatistics {
  loaded_units: number;
  cleaned_units: number;
  chunk_count: number;
  total_characters: number;
}

export interface DocumentUploadResponse {
  status: string;
  document_id: string;
  file_name: string;
  file_type: string;
  statistics: IngestionStatistics;
}

export interface IndexResponse {
  status: string;
  document_id: string;
  collection_name: string;
  indexed_chunks: number;
  embedding_model: string;
  embedding_dimension: number;
}

export interface SessionDocument extends DocumentUploadResponse {
  indexState: "uploaded" | "indexing" | "indexed" | "error";
  indexError?: string;
  indexResult?: IndexResponse;
}

export interface RagSource {
  source_number: number;
  chunk_id: string;
  document_id?: string | null;
  text: string;
  distance: number;
  score?: number | null;
  distance_metric: string;
  metadata: Record<string, unknown>;
}

export interface RagResponse { answer: string; model: string; sources: RagSource[] }
