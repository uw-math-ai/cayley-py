import { expect, test } from "@playwright/test";

async function assertNoHorizontalOverflow(page: import("@playwright/test").Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(2);
}

test("renders graph canvas and responsive shell", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("explorer-canvas")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Explorer" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Challenge" })).toBeVisible();
  await expect(page.getByText(/nodes$/)).toBeVisible();
  await expect(page.getByText(/edges$/)).toBeVisible();
  await assertNoHorizontalOverflow(page);

  const nonBlank = await page.getByTestId("explorer-canvas").evaluate((canvas) => {
    const target = canvas as HTMLCanvasElement;
    const gl = target.getContext("webgl", { preserveDrawingBuffer: true });
    if (!gl || target.width === 0 || target.height === 0) return false;
    const pixels = new Uint8Array(4);
    gl.readPixels(Math.floor(target.width / 2), Math.floor(target.height / 2), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return Array.from(pixels).some((value) => value !== 0);
  });
  expect(nonBlank).toBe(true);
});

test("clicking a node highlights a certified shortest path", async ({ page }) => {
  await page.goto("/");
  const canvas = page.getByTestId("explorer-canvas");
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.click(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await expect(page.getByTestId("selected-path")).toBeVisible();
  await expect(page.getByText("Selected")).toBeVisible();
  await expect(page.getByText("Length")).toBeVisible();
  await expect(page.getByText("Word")).toBeVisible();
  await page.mouse.click(box!.x + 12, box!.y + 12);
  await expect(page.getByTestId("selected-path")).toBeHidden();
  await assertNoHorizontalOverflow(page);
});

test("challenge can start and reveal certified forfeit", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("tab", { name: "Challenge" }).click();
  const challengeSetup = page.locator(".challenge-setup");
  await challengeSetup.getByLabel("Family").selectOption("lrx");
  await expect(challengeSetup.getByRole("spinbutton", { name: "n" })).toBeVisible();
  await expect(challengeSetup.getByLabel("Graph")).toBeVisible();
  await expect(challengeSetup.getByLabel("Inverses")).toBeVisible();
  await expect(challengeSetup.getByLabel("Layout")).toBeVisible();
  await expect(challengeSetup.getByLabel("Level")).toBeVisible();
  await challengeSetup.getByRole("button", { name: "New" }).click();
  await expect(page.getByRole("heading", { name: "User" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "BFS" })).toBeVisible();
  const objective = page.getByTestId("challenge-objective");
  await expect(objective).toBeVisible();
  await expect(page.getByText("Start")).toBeVisible();
  await expect(objective.getByText("Goal")).toBeVisible();
  await expect(page.getByText("Score")).toBeVisible();
  await expect(page.getByTestId("score-card")).toBeVisible();
  await expect(page.getByTestId("objective-user-word")).toHaveText("id");
  await expect(page.getByTestId("objective-bfs-word")).toHaveText("id");
  await expect(page.getByTestId("user-word")).toHaveText("id");
  await expect(page.getByTestId("bfs-word")).toHaveText("id");
  await expect(page.getByText("Status")).toBeVisible();
  const legalMoves = page.getByTestId("legal-moves");
  await expect(page.getByText(/moves$/)).toBeVisible();
  await expect(legalMoves.getByText("Moves")).toBeVisible();
  await expect(legalMoves.getByText("L", { exact: true })).toBeVisible();
  await expect(legalMoves.getByText("R", { exact: true })).toBeVisible();
  await expect(legalMoves.getByText("X", { exact: true })).toBeVisible();
  await expect(page.getByText(/legal/i)).toHaveCount(0);
  await expect(page.getByTestId("challenge-canvas")).toBeVisible();
  await expect(page.getByTestId("challenge-bfs-canvas")).toBeVisible();
  await expect(page.getByTestId("challenge-canvas-goal-marker")).toBeVisible();
  await expect(page.getByRole("button", { name: "L", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "R", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "X", exact: true })).toHaveCount(0);
  await challengeSetup.getByRole("button", { name: "Give up" }).click();
  await expect(page.getByText("Shortest")).toBeVisible();
  await assertNoHorizontalOverflow(page);
});

test("bruhat layout and k-different schreier view load", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Layout").selectOption("bruhat");
  await page.getByLabel("Graph").selectOption("k_different");
  await page.getByLabel("different k").fill("3");
  await page.getByRole("button", { name: "View" }).click();
  await expect(page.getByText("k_different")).toBeVisible();
  await expect(page.getByText(/edges$/)).toBeVisible();
  await assertNoHorizontalOverflow(page);
});

test("new explorer layouts load", async ({ page }) => {
  await page.goto("/");
  for (const layout of ["spectral", "lehmer", "coset", "target-distance"]) {
    await page.getByLabel("Layout").selectOption(layout);
    await page.getByRole("button", { name: "View" }).click();
    await expect(page.locator(".stats").getByText(layout === "target-distance" ? "target-distance" : layout)).toBeVisible();
  }
  await assertNoHorizontalOverflow(page);
});
