import { test, expect } from '@playwright/test';

test.describe('Homepage visual regression', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for the page content to fully load
    await page.waitForSelector('text=Welcome to MLflow');
  });

  test('Getting Started section renders correctly', async ({ page }) => {
    // Dismiss the telemetry banner if present, so it doesn't affect screenshots
    const dismissButton = page.locator('[aria-label="Close"]').first();
    if (await dismissButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await dismissButton.click();
      await dismissButton.waitFor({ state: 'hidden' });
    }

    // Ensure the Getting Started section is expanded and feature cards are visible
    await expect(page.getByText('Getting Started')).toBeVisible();
    await expect(page.getByText('Tracing')).toBeVisible();

    const gettingStarted = page.locator('section').filter({ hasText: 'Getting Started' });
    await expect(gettingStarted).toHaveScreenshot('getting-started-section.png');
  });

  test('feature cards have consistent styling', async ({ page }) => {
    // Dismiss the telemetry banner if present
    const dismissButton = page.locator('[aria-label="Close"]').first();
    if (await dismissButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await dismissButton.click();
      await dismissButton.waitFor({ state: 'hidden' });
    }

    // Wait for all feature cards to render
    const featureSection = page.locator('section').filter({ hasText: 'Getting Started' });
    await expect(featureSection.getByText('Tracing')).toBeVisible();
    await expect(featureSection.getByText('Model Training')).toBeVisible();

    // Screenshot the feature cards row
    const featureCards = page.locator('section').filter({ hasText: 'Getting Started' }).locator('> div > div:last-child');
    await expect(featureCards).toHaveScreenshot('feature-cards.png');
  });

  test('full homepage layout', async ({ page }) => {
    // Dismiss the telemetry banner if present
    const dismissButton = page.locator('[aria-label="Close"]').first();
    if (await dismissButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await dismissButton.click();
      await dismissButton.waitFor({ state: 'hidden' });
    }

    // Wait for experiments table to load
    await expect(page.getByText('Recent Experiments')).toBeVisible();

    // Full page screenshot
    await expect(page).toHaveScreenshot('homepage-full.png', {
      fullPage: true,
    });
  });
});
