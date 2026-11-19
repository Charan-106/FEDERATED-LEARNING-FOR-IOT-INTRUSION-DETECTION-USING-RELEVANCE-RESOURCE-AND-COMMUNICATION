/**
 * Test script to verify API behavior after rollback and dataset audit integration
 */

const http = require('http');

function request(options, body) {
  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, headers: res.headers, body: JSON.parse(data) });
        } catch (e) {
          resolve({ status: res.statusCode, headers: res.headers, raw: data });
        }
      });
    });
    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

async function main() {
  const loginRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/auth/login',
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }, { email: 'admin@arc-fl.local', password: 'Admin@123' });

  const token = loginRes.body.token;
  console.log('Logged in, got token:', !!token);

  console.log('\n--- 1b. Testing GET /api/models/current with Auth ---');
  const currentModelRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/models/current',
    method: 'GET',
    headers: { 'Authorization': `Bearer ${token}` },
  });
  console.log('Status:', currentModelRes.status);
  console.log('Active model:', currentModelRes.body.model?.version, 'Checksum:', currentModelRes.body.model?.checksumSha256);
  console.log('Baseline Accuracy:', currentModelRes.body.model?.baselineAccuracy);

  console.log('\n--- 3. Testing GET /api/admin/dashboard ---');
  const dashRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/admin/dashboard',
    method: 'GET',
    headers: { 'Authorization': `Bearer ${token}` },
  });
  console.log('Dashboard active model:', JSON.stringify(dashRes.body.activeModel));

  console.log('\n--- 4. Testing POST /api/admin/dataset-audit ---');
  const auditRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/admin/dataset-audit',
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
  }, { clientIds: ['client-001', 'client-002'] });

  console.log('Audit total classes expected:', auditRes.body.totalExpectedClasses);
  console.log('Missing across all clients count:', auditRes.body.missingAcrossAllClients?.length);
  console.log('Missing classes across all clients:', auditRes.body.missingAcrossAllClients);
  console.log('Audit warnings count:', auditRes.body.warnings?.length);
  console.log('First 2 warnings:');
  console.log(auditRes.body.warnings?.slice(0, 2));

  console.log('\n--- 5. Testing GET /api/admin/audit-logs (latest 3) ---');
  const logsRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/admin/audit-logs?limit=3',
    method: 'GET',
    headers: { 'Authorization': `Bearer ${token}` },
  });
  console.log('Latest audit logs:');
  logsRes.body.logs?.slice(0, 3).forEach(l => {
    console.log(`- Action: ${l.action}, Resource: ${l.resourceType} (${l.resourceId}), Reason: ${l.detailsJson?.reason || 'N/A'}`);
  });
}

main().catch(console.error);
