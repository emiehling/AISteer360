import { apiFetch } from "./client";
import type { CatalogEntry, CatalogResponse } from "../types";

export async function fetchCatalog(): Promise<CatalogEntry[]> {
  const data = await apiFetch<CatalogResponse>("/api/catalog");
  return data.entries;
}
