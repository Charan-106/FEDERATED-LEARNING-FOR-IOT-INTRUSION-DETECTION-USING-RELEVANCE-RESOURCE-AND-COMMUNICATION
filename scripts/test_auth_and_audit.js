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
  console.log('=== VERIFYING ADMIN AUTHORIZATION ON DATASET-AUDIT ENDPOINTS ===\n');

  // Test 1: Unauthenticated POST /api/admin/dataset-audit
  const unauthRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/admin/dataset-audit',
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }, { clientIds: ['client-001'] });
  console.log('1. Unauthenticated POST /api/admin/dataset-audit -> Status:', unauthRes.status, unauthRes.body);

  // Test 2: Unauthenticated GET /api/admin/rounds/some-round-id/dataset-audit
  const unauthGetRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/admin/rounds/some-round-id/dataset-audit',
    method: 'GET',
  });
  console.log('2. Unauthenticated GET /api/admin/rounds/.../dataset-audit -> Status:', unauthGetRes.status, unauthGetRes.body);

  // Test 3: Authenticated as CLIENT (Edge Node Alpha)
  const clientLogin = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/auth/login',
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }, { email: 'client1@arc-fl.local', password: 'Client@123' });
  const clientToken = clientLogin.body.token;

  const clientPostRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/admin/dataset-audit',
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${clientToken}` },
  }, { clientIds: ['client-001'] });
  console.log('3. Client role POST /api/admin/dataset-audit -> Status:', clientPostRes.status, clientPostRes.body);

  // Test 4: Authenticated as ADMIN
  const adminLogin = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/auth/login',
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }, { email: 'admin@arc-fl.local', password: 'Admin@123' });
  const adminToken = adminLogin.body.token;

  const adminPostRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/admin/dataset-audit',
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${adminToken}` },
  }, { clientIds: ['client-001', 'client-002'] });
  console.log('4. Admin role POST /api/admin/dataset-audit -> Status:', adminPostRes.status);
  console.log('   Expected classes:', adminPostRes.body.totalExpectedClasses);
  console.log('   Missing across all:', adminPostRes.body.missingAcrossAllClients);
  console.log('   Per-client audit:');
  for (const [cid, c] of Object.entries(adminPostRes.body.clients || {})) {
    console.log(`   * ${cid} (${c.datasetFile}): ${c.sampleCount} samples, ${c.uniqueClassCount} classes present, ${c.missingClasses?.length} classes missing`);
  }
  console.log('   Warnings:', adminPostRes.body.warnings);

  // Test 5: Verify Model API and Admin Dashboard both report initial-v1
  console.log('\n=== VERIFYING MODEL API & ADMIN DASHBOARD ===\n');
  const modelApiRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/models/current',
    method: 'GET',
    headers: { 'Authorization': `Bearer ${adminToken}` },
  });
  console.log('Model API (/api/models/current):');
  console.log('   Version:', modelApiRes.body.model?.version);
  console.log('   ID:', modelApiRes.body.model?.id);
  console.log('   isActive:', modelApiRes.body.model?.isActive);
  console.log('   Checksum:', modelApiRes.body.model?.checksumSha256);
  console.log('   Baseline Accuracy:', modelApiRes.body.model?.baselineAccuracy);

  const dashRes = await request({
    hostname: 'localhost',
    port: 5000,
    path: '/api/admin/dashboard',
    method: 'GET',
    headers: { 'Authorization': `Bearer ${adminToken}` },
  });
  console.log('Admin Dashboard (/api/admin/dashboard):');
  console.log('   Active Model:', dashRes.body.activeModel);
}

main().catch(console.error);
