// Verify the actual project browser and live agent execution in Microsoft Edge.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('../.cache/recording/node_modules/playwright-core');

const output = path.resolve('results/data-browser/browser');
fs.mkdirSync(output, { recursive: true });
const retail = JSON.parse(fs.readFileSync('../.dependencies/tau2-033bf082f48e/data/tau2/domains/retail/db.json', 'utf8'));
const orderIds = Object.keys(retail.orders).sort();
const box = (page, label) => page.getByTestId('stSelectbox').filter({ has: page.getByText(label, { exact: true }) }).getByRole('combobox');
async function choose(page, label, option) {
  await box(page, label).click();
  await page.getByRole('option', { name: option, exact: true }).click();
}
async function main() {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  let active;
  try {
    for (const width of [360, 768, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 1050 } });
      const page = await context.newPage(); active = page;
      await page.goto('http://127.0.0.1:8501');
      await page.getByRole('heading', { name: 'CasePilot Workspace', exact: true }).waitFor();
      await page.getByText('1,000 / 1,000 records · select a row to view details', { exact: true }).waitFor();
      assert.equal(await page.getByText('Ca mẫu & tra cứu', { exact: true }).count(), 0);
      const search = page.getByRole('textbox', { name: 'Search table', exact: true });
      await search.fill('not-present@example.test'); await search.press('Enter');
      await page.getByText('Không có bản ghi phù hợp với bộ lọc.', { exact: true }).waitFor();
      await search.fill(''); await search.press('Enter');
      await page.getByRole('button', { name: 'Attach record to chat', exact: true }).waitFor();
      await choose(page, 'Bảng', 'Khách hàng');
      await page.getByText('500 / 500 records · select a row to view details', { exact: true }).waitFor();
      await choose(page, 'Bảng', 'Đơn hàng');
      await page.getByText('1,000 / 1,000 records · select a row to view details', { exact: true }).waitFor();
      await page.getByRole('button', { name: 'Attach record to chat', exact: true }).waitFor();
      const grid = page.getByTestId('stDataFrame').first();
      await grid.locator('.dvn-scroller').first().click({ position: { x: 16, y: 81 } });
      await page.getByText('Orders / ' + orderIds[1], { exact: true }).waitFor();
      await choose(page, 'Record', orderIds[2]);
      await page.getByText('Orders / ' + orderIds[2], { exact: true }).waitFor();
      await page.getByRole('button', { name: 'Attach record to chat', exact: true }).waitFor();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      assert.equal(overflow, false, 'Page overflow at ' + width);
      await page.getByRole('button', { name: 'New session', exact: true }).focus();
      await page.keyboard.press('Tab');
      assert.notEqual(await page.evaluate(() => document.activeElement.tagName), 'BODY');
      await page.screenshot({ path: path.join(output, `database-${width}.png`), fullPage: true });
      if (width <= 768) {
        await page.locator('.st-key-mobile_switch').getByText('Agent', { exact: true }).click();
        await page.getByTestId('stChatInputTextArea').waitFor({ state: 'visible' });
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
        await page.screenshot({ path: path.join(output, `agent-${width}.png`), fullPage: true });
      }
      results.push({ width, page_overflow: overflow, complete_database: true, filters: true, row_selection: true, mobile_switch: width <= 768 });
      await context.close();
    }
    if (process.argv.includes('--research')) {
      const page = await browser.newPage(); active = page;
      await page.goto('http://127.0.0.1:8502');
      await page.getByRole('heading', { name: 'CasePilot Lite', exact: true }).waitFor();
      results.push({ research_entrypoint: true });
      await page.close();
    }
    if (process.argv.includes('--live')) {
      const db = JSON.parse(fs.readFileSync('../.dependencies/tau2-033bf082f48e/data/tau2/domains/retail/db.json', 'utf8'));
      const order = Object.values(db.orders).find(o => o.status === 'pending');
      const context = await browser.newContext({ viewport: { width: 1440, height: 1050 },
        recordVideo: { dir: output, size: { width: 1440, height: 1050 } } });
      const page = await context.newPage(); active = page;
      page.setDefaultTimeout(180000);
      await page.goto('http://127.0.0.1:8501');
      const input = page.getByTestId('stChatInputTextArea');
      const search = page.getByRole('textbox', { name: 'Search table', exact: true });
      await search.fill(order.order_id); await search.press('Enter');
      await page.getByText('1 / 1,000 records · select a row to view details', { exact: true }).waitFor();
      await page.getByRole('button', { name: 'Attach record to chat', exact: true }).click();
      await page.getByText('Attached: orders / ' + order.order_id, { exact: true }).waitFor();
      await input.fill('Bạn xác minh và xem chi tiết đơn đã đính kèm giúp tôi.'); await input.press('Enter');
      await page.getByText('Model · Model · Processing', { exact: true }).first().waitFor();
      await page.screenshot({ path: path.join(output, 'live-running.png'), fullPage: true });
      console.log('Observed live model activity before reply');
      await page.getByText(/Agent activity · turn 1 ·/).waitFor();
      await input.fill(`Tôi muốn hủy đơn ${order.order_id} vì không còn nhu cầu. Chuẩn bị yêu cầu giúp tôi nhé.`);
      await input.press('Enter');
      await page.getByRole('button', { name: 'Confirm action', exact: true }).waitFor();
      await page.screenshot({ path: path.join(output, 'live-confirmation.png'), fullPage: true });
      await page.getByRole('button', { name: 'Confirm action', exact: true }).click();
      await page.getByText(order.order_id + ' · Cancelled', { exact: true }).waitFor();
      await page.screenshot({ path: path.join(output, 'live-database-updated.png'), fullPage: true });
      await page.getByText(/Agent activity · turn 3 ·/).waitFor();
      await page.getByText(/Tool · cancel_pending_order · Completed/).waitFor();
      await page.getByText('Order status: Cancelled', { exact: true }).waitFor();
      await page.getByRole('tab', { name: 'JSON', exact: true }).click();
      await page.screenshot({ path: path.join(output, 'live-completed.png'), fullPage: true });
      results.push({ live: true, attached_actual_record: true, model_running_before_reply: true,
        confirmation: true, completed_tool: 'cancel_pending_order', session_source: true,
        database_updated_before_final_reply: true });
      await context.close();
      fs.copyFileSync(await page.video().path(), path.join(output, 'data-browser-agent.webm'));
    }
    const report = process.argv.includes('--live') ? 'validation-live.json' : 'validation.json';
    fs.writeFileSync(path.join(output, report), JSON.stringify(results, null, 2));
    console.log(JSON.stringify(results));
  } catch (error) {
    if (active && !active.isClosed()) {
      await active.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
      fs.writeFileSync(path.join(output, 'failure.txt'), await active.locator('body').innerText());
    }
    throw error;
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
