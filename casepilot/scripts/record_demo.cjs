// Record the real local dashboard and saved artifacts, without issuing provider requests.
// Setup: npm install --prefix .cache/recording --no-audit --no-fund playwright-core@1.64.1
// Then: node .cache/recording/node_modules/playwright-core/cli.js install ffmpeg
// Run: node scripts/record_demo.cjs
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('../.cache/recording/node_modules/playwright-core');

async function main() {
  const url = process.env.CASEPILOT_DEMO_URL || 'http://127.0.0.1:8502';
  if (!['localhost', '127.0.0.1'].includes(new URL(url).hostname)) {
    throw new Error('Demo recording requires the local dashboard');
  }
  const output = path.resolve('results/demo-video');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true,
    args: ['--disable-gpu', '--no-first-run'] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 },
    recordVideo: { dir: output, size: { width: 1440, height: 900 } } });
  const page = await context.newPage();
  const stages = [];
  async function pause(stage, milliseconds = 2500) {
    console.log('Recording stage: ' + stage);
    stages.push({ stage, at: new Date().toISOString() });
    await page.screenshot({ path: path.join(output, `${stages.length}-${stage}.png`) });
    await page.waitForTimeout(milliseconds);
  }
  async function openSaved(run) {
    await page.goto(url + '/?saved_run=' + encodeURIComponent(run), {
      waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.getByRole('combobox', { name: 'Episode', exact: true }).waitFor();
  }
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.getByRole('heading', { name: 'CasePilot Lite', exact: true }).waitFor();
    await page.getByRole('combobox', { name: 'Task', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Run simulation', exact: true }).waitFor();
    await pause('simulation-controls');
    await page.getByText('Saved runs — inspect without API calls', { exact: true }).click();
    const run = process.env.CASEPILOT_DEMO_RUN || path.join('results', 'demo-mcp-f2',
      '4065cee4e8774e84ad6aee033fb37021');
    await openSaved(run);
    await page.getByText('Evaluator reward', { exact: true }).scrollIntoViewIfNeeded();
    await pause('mcp-f2-reward');
    for (const label of ['faults', 'operations', 'agent']) {
      const expander = page.getByTestId('stExpander').filter({
        has: page.getByText(label, { exact: true }) });
      await expander.locator('summary').click();
      await expander.scrollIntoViewIfNeeded();
      await pause(`mcp-${label}`);
    }
    await page.getByRole('tab', { name: 'Evaluation', exact: true }).click();
    await page.getByRole('checkbox', { name: 'Include earlier attempts' }).waitFor();
    await page.getByRole('checkbox', { name: 'Include earlier attempts' }).scrollIntoViewIfNeeded();
    await pause('evaluation-comparison', 3500);
    await page.getByText('Retrieval component metrics', { exact: true }).click();
    await page.getByText('Retrieval component metrics', { exact: true }).scrollIntoViewIfNeeded();
    await pause('retrieval-component');
    await context.close();
    const videoPath = await page.video().path();
    const finalPath = path.join(output, 'casepilot-dashboard.webm');
    fs.copyFileSync(videoPath, finalPath);
    fs.writeFileSync(path.join(output, 'recording.json'), JSON.stringify({
      recorded_at: new Date().toISOString(), type: 'actual local browser recording',
      note: 'Saved MCP F2 example has reward 0; one receipt reconciliation and zero duplicate commits. '
        + 'No simulation button was clicked. Experiments may still be running.',
      stages, video: path.basename(finalPath)
    }, null, 2));
    console.log('Recording saved: results/demo-video/casepilot-dashboard.webm');
  } catch (error) {
    await page.screenshot({ path: path.join(output, 'recording-error.png') });
    throw error;
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
