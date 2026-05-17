import { apiFetch } from "./client";
import type { MethodsResponse, MethodSpec } from "../types";

export async function fetchMethods(): Promise<MethodSpec[]> {
  const data = await apiFetch<MethodsResponse>("/api/methods");
  return data.methods;
}
