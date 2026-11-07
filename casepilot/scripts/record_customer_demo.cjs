// Actual localhost UI capture. --live explicitly enables provider calls and synthetic cancellation.
// npm.cmd install --prefix .cache/recording --no-audit --no-fund playwright-core@1.64.1
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('../.cache/recording/node_modules/playwright-core');

async function main() {
  if (!process.argv.includes('--live')) throw new Error('Pass --live: uses the model API and cancels one synthetic order.');
  const url = process.env.CASEPILOT_DEMO_URL || 'http://127.0.0.1:8501';
  if (!['localhost', '127.0.0.1'].includes(new URL(url).hostname)) throw new Error('Localhost only');
  const root = path.resolve('results/customer-demo', new Date().toISOString().replace(/[:.]/g, '-'));
  fs.mkdirSync(root, { recursive: true });
  const db = JSON.parse(fs.readFileSync('../.dependencies/tau2-033bf082f48e/data/tau2/domains/retail/db.json', 'utf8'));
  const order = Object.values(db.orders).find(row => row.status === 'pending');
  assert.ok(order, 'A pending synthetic order is required');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(180000);
  const report = { source: 'Actual Streamlit browser screenshots; no mocked model/tool responses',
    recorded_at: new Date().toISOString(), synthetic: true, order_id: order.order_id, clips: [] };
  let clip;
  let recording = true;
  let failure;
  async function frame() {
    if (!clip || page.isClosed()) return;
    const current = clip;
    const name = `${String(current.frames.length).padStart(4, '0')}.png`;
    await page.screenshot({ path: path.join(root, current.name, name), timeout: 15000 });
    current.frames.push({ file: name, at_ms: Date.now() - current.started });
  }
  async function start(name) {
    clip = { name, started: Date.now(), frames: [] };
    fs.mkdirSync(path.join(root, name));
    report.clips.push(clip);
    console.log('Recording ' + name);
  }
  const capture = (async () => {
    while (recording) {
      try { await frame(); } catch (error) { failure = error; }
      await new Promise(resolve => setTimeout(resolve, 900));
    }
  })();
  async function hold(ms = 2200) { await page.waitForTimeout(ms); }
  async function followActivity() {
    const status = page.getByText('Agent is working…', { exact: true }).last();
    if (await status.isVisible()) await status.scrollIntoViewIfNeeded({ timeout: 5000 });
  }
  async function send(text) {
    const input = page.getByTestId('stChatInputTextArea');
    await input.fill(text); await hold(1500); await input.press('Enter');
    await page.getByText('Model · Model · Processing', { exact: true }).first().waitFor();
    await followActivity(); await hold(1500);
  }
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: 'Project data', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Attach record to chat', exact: true }).waitFor();
    const dismiss = page.getByRole('button', { name: "Don't show again", exact: true });
    if (await dismiss.isVisible()) await dismiss.click();
    await start('01-data-workspace'); await hold(2500);
    const search = page.getByRole('textbox', { name: 'Search table', exact: true });
    await search.fill(order.order_id); await search.press('Enter');
    await page.getByText('1 / 1,000 records · select a row to view details', { exact: true }).waitFor();
    await hold();
    await page.getByRole('button', { name: 'Attach record to chat', exact: true }).click();
    await page.getByText('Attached: orders / ' + order.order_id, { exact: true }).waitFor();
    await hold();

    await start('02-agent-tools');
    await send(`My email is ${db.users[order.user_id].email}. Please show order ${order.order_id}.`);
    await page.getByText(/Agent activity · turn 1 ·/).waitFor();
    const activity = page.getByTestId('stExpander').filter({ hasText: /Agent activity · turn 1 ·/ }).first();
    await activity.scrollIntoViewIfNeeded();
    assert.ok(await page.getByText(/Tool · get_order_details · Completed/).count());
    await hold(3000);
    const details = activity.locator('summary').filter({ hasText: 'Input / result' });
    if (await details.count()) { await details.last().click(); await details.last().scrollIntoViewIfNeeded(); await hold(2500); }

    await start('03-confirmation');
    await send(`Please prepare cancellation of order ${order.order_id}; I no longer need it. The reason is "no longer needed".`);
    const confirm = page.getByRole('button', { name: 'Confirm action', exact: true });
    await confirm.waitFor(); await confirm.scrollIntoViewIfNeeded(); await hold(3500);
    report.confirmation_observed_before_execution = true;
    await confirm.click();
    await page.getByText(/Tool · cancel_pending_order · Completed/).waitFor();
    await page.getByText(/Agent activity · turn 3 ·/).waitFor();
    const final = page.getByTestId('stExpander').filter({ hasText: /Agent activity · turn 3 ·/ }).first();
    await final.scrollIntoViewIfNeeded(); await hold(2500);
    // Show the current-session record as independent UI evidence of the mutation.
    await search.scrollIntoViewIfNeeded();
    await page.getByText('Order status: Cancelled', { exact: true }).waitFor();
    await page.getByText('Order status: Cancelled', { exact: true }).scrollIntoViewIfNeeded();
    await hold(3000);
    report.cancel_tool_completed = true;
    report.current_session_order_cancelled = true;
    if (failure) throw failure;
    report.status = 'passed';
  } catch (error) {
    report.status = 'failed';
    await page.screenshot({ path: path.join(root, 'failure.png'), fullPage: true }).catch(() => {});
    fs.writeFileSync(path.join(root, 'failure.txt'), await page.locator('body').innerText().catch(() => ''));
    throw error;
  } finally {
    recording = false; await capture;
    fs.writeFileSync(path.join(root, 'recording.json'), JSON.stringify(report, null, 2));
    await browser.close();
    console.log('Capture: ' + root);
  }
}
main().catch(error => { console.error(error.stack); process.exitCode = 1; });
