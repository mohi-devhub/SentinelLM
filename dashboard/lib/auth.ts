const STORAGE_KEY = "sentinellm_api_key";
const SKIP_KEY = "sentinellm_skip_key";

export function getApiKey(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(STORAGE_KEY);
}

export function setApiKey(key: string): void {
  window.localStorage.setItem(STORAGE_KEY, key);
  window.localStorage.removeItem(SKIP_KEY);
}

export function clearApiKey(): void {
  window.localStorage.removeItem(STORAGE_KEY);
  window.localStorage.removeItem(SKIP_KEY);
}

/** Whether the user chose "Continue without a key" — persisted so the gate
 * doesn't reappear on every reload for deployments that don't need auth. */
export function getSkippedKey(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(SKIP_KEY) === "true";
}

export function setSkippedKey(): void {
  window.localStorage.setItem(SKIP_KEY, "true");
}
