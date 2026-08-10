import { expect, test } from "@playwright/test";

test("the TypeScript showcase executes its generated Tach kernel", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto("/");
  await expect(page.locator("#status")).toHaveText("Integrated 2 particles on WebGPU.");

  const particles = JSON.parse(await page.locator("#output").textContent());
  expect(particles).toEqual([
    { position: [2, 4, 6, 1], velocity: [2, 4, 6, 0] },
    { position: [-0.5, -1, -1.5, 1], velocity: [1, 2, 3, 0] },
  ]);
  expect(pageErrors).toEqual([]);
});
