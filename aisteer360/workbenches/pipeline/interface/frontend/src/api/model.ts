import { apiFetch } from "./client";
import type { ModelProbe } from "../types";

export async function probeModel(modelId: string): Promise<ModelProbe> {
  const qs = new URLSearchParams({ model_id: modelId });
  return apiFetch<ModelProbe>(`/api/model/probe?${qs.toString()}`);
}

export interface ModelSearchHit {
  model_id: string;
  downloads: number | null;
  likes: number | null;
}

export interface ModelSearchResponse {
  query: string;
  results: ModelSearchHit[];
}

export async function searchModels(query: string, limit = 10): Promise<ModelSearchResponse> {
  const qs = new URLSearchParams({ q: query, limit: String(limit) });
  return apiFetch<ModelSearchResponse>(`/api/model/search?${qs.toString()}`);
}
