import { defineConfig } from "@playwright/test";
import base from "./playwright.config.mjs";

export default defineConfig(base, { metadata: { profile: "gpu-large" } });
