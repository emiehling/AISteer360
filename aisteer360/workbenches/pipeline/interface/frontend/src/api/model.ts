import { apiFetch } from "./client";
import type { ModelProbe } from "../types";

export async function probeModel(modelId: string): Promise<ModelProbe> {
  const qs = new URLSearchParams({ model_id: modelId });
  return apiFetch<ModelProbe>(`/api/model/probe?${qs.toString()}`);
}
