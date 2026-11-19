const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { PrismaClient } = require(path.resolve(__dirname, '../backend/node_modules/@prisma/client'));

const prisma = new PrismaClient();

function getFileHash(filePath) {
  if (!fs.existsSync(filePath)) return null;
  const buf = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buf).digest('hex');
}

async function verify() {
  console.log('=== ARC-FL FINAL READ-ONLY RECOVERY VERIFICATION ===\n');

  // 1. Active Model in DB
  const allModels = await prisma.globalModel.findMany({
    orderBy: { createdAt: 'desc' },
  });
  console.log(`Total Global Models in DB: ${allModels.length}`);
  allModels.forEach((m) => {
    console.log(`- Version: ${m.version} | ID: ${m.id} | isActive: ${m.isActive} | Checksum: ${m.checksumSha256}`);
  });

  const activeModels = allModels.filter((m) => m.isActive);
  console.log(`\nActive Models Count: ${activeModels.length}`);
  const activeModel = activeModels[0];
  if (activeModel) {
    console.log(`Active Model: ${activeModel.version} (${activeModel.id})`);
  }

  // 1b. initial-v1 checkpoint hash
  const initV1DiskPath = path.resolve(__dirname, '../storage/global-models/initial-v1/best.pt');
  const initV1Hash = getFileHash(initV1DiskPath);
  console.log(`initial-v1 disk path: ${initV1DiskPath}`);
  console.log(`initial-v1 disk SHA-256: ${initV1Hash}`);

  // 2. Historical model global-v22
  const v22Model = allModels.find((m) => m.version === 'global-v22');
  console.log(`\nglobal-v22 in DB: ${v22Model ? 'FOUND' : 'MISSING'}`);
  if (v22Model) {
    console.log(`global-v22 isActive: ${v22Model.isActive}`);
  }
  const v22DiskPath = path.resolve(__dirname, '../storage/global-models/global-v22/best.pt');
  const v22Hash = getFileHash(v22DiskPath);
  console.log(`global-v22 disk path: ${v22DiskPath}`);
  console.log(`global-v22 disk SHA-256: ${v22Hash}`);

  // 3. Candidate metadata for global-v22
  const candidateV22Path = path.resolve(__dirname, '../storage/candidate-models/candidate-v22.json');
  console.log(`candidate-v22.json exists: ${fs.existsSync(candidateV22Path)}`);

  // 4. Audit Log for Rollback
  const rollbackLogs = await prisma.auditLog.findMany({
    where: { action: 'ROLLBACK_MODEL' },
    orderBy: { timestamp: 'desc' },
  });
  console.log(`\nRollback Audit Logs Count: ${rollbackLogs.length}`);
  rollbackLogs.forEach((l) => {
    console.log(`- Time: ${l.createdAt} | Action: ${l.action} | Resource: ${l.resourceType} (${l.resourceId})`);
    console.log(`  Details:`, JSON.stringify(l.detailsJson, null, 2));
  });
}

verify()
  .catch(console.error)
  .finally(async () => {
    await prisma.$disconnect();
  });
