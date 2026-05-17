import { apiFetch } from "./client";
import type { PipelineDefinition } from "../types";

export interface SessionSummary {
  id: string;
  model_name: string;
  status: string;
  error: string | null;
  stale: boolean;
  created_at: number;
  updated_at: number;
  last_heartbeat: number | null;
  idle_timeout_s: number;
}

export interface AgentCommand {
  command: string;
  server: string;
  session_id: string;
  agent_token: string;
}

export interface SessionCreateResponse {
  session: SessionSummary;
  agent_token: string;
  agent_command: AgentCommand;
  dispatch_status: "local" | "ssh" | "manual" | "failed";
  dispatch_error: string | null;
}

export interface InferenceAcceptedResponse {
  request_id: string;
}

export interface InferRequest {
  pipeline: PipelineDefinition;
  prompt: string;
  gen_kwargs?: Record<string, unknown>;
  runtime_kwargs?: Record<string, unknown>;
  request_id?: string;
}

export async function createSession(modelNameOrPath: string): Promise<SessionCreateResponse> {
  return apiFetch<SessionCreateResponse>("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ model_name_or_path: modelNameOrPath }),
  });
}

export async function getSession(id: string): Promise<SessionSummary> {
  return apiFetch<SessionSummary>(`/api/sessions/${id}`);
}

export async function infer(id: string, body: InferRequest): Promise<InferenceAcceptedResponse> {
  return apiFetch<InferenceAcceptedResponse>(`/api/sessions/${id}/infer`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
