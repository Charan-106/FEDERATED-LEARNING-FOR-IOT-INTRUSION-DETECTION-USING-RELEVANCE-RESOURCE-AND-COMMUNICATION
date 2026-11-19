/**
 * Script to safely execute the model rollback from global-v22 to initial-v1
 * Preserves global-v22 records, checkpoint, candidate metadata, and diagnostic reports.
 * Updates initial-v1 to active with verified baseline accuracy (95.20% on Benchmark A).
 * Logs ROLLBACK_MODEL in AuditLog with reason 'severe validation degradation'.
 */

const path = require('path');
const { PrismaClient } = require(path.resolve(__dirname, '../backend/node_modules/@prisma/client'));
const fs = require('fs');
const crypto = require('crypto');

const prisma = new PrismaClient();

function computeSha256(filePath) {
  const buffer = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

async function main() {
  console.log('=== ARC-FL MODEL ROLLBACK EXECUTION ===');

  // 1. Verify models exist
  const initialV1 = await prisma.globalModel.findUnique({
    where: { id: '7ff1f170-b1b5-48a5-8297-bc0392b704b8' },
  });
  if (!initialV1) throw new Error('initial-v1 record not found in database!');

  const globalV22 = await prisma.globalModel.findUnique({
    where: { id: 'ff9f7bea-3318-47c2-9355-7bf053137364' },
  });
  if (!globalV22) throw new Error('global-v22 record not found in database!');

  console.log(`Current active model: ${globalV22.isActive ? 'global-v22' : initialV1.isActive ? 'initial-v1' : 'None'}`);

  // 2. Verify physical files on disk
  const basePath = path.resolve(__dirname, '..', 'storage', 'global-models');
  const initialV1File = path.join(basePath, 'initial-v1', 'best.pt');
  const globalV22File = path.join(basePath, 'global-v22', 'best.pt');

  if (!fs.existsSync(initialV1File)) throw new Error(`initial-v1 file not found at ${initialV1File}`);
  if (!fs.existsSync(globalV22File)) throw new Error(`global-v22 file not found at ${globalV22File}`);

  const initialV1Hash = computeSha256(initialV1File);
  const globalV22Hash = computeSha256(globalV22File);

  console.log(`initial-v1 SHA-256 on disk: ${initialV1Hash}`);
  console.log(`Expected initial-v1 SHA-256: 57b7afd9d01c30b48101fc2660d699ea1b5790104c00e534e00e090b517d4993`);
  if (initialV1Hash !== '57b7afd9d01c30b48101fc2660d699ea1b5790104c00e534e00e090b517d4993') {
    throw new Error('initial-v1 checksum does not match expected original hash!');
  }

  console.log(`global-v22 SHA-256 on disk: ${globalV22Hash}`);

  // 3. Find admin user
  const adminUser = await prisma.user.findFirst({ where: { role: 'ADMIN' } });
  if (!adminUser) throw new Error('Admin user not found!');

  // 4. Execute atomic transaction
  const result = await prisma.$transaction(async (tx) => {
    // Deactivate all models
    await tx.globalModel.updateMany({
      data: { isActive: false },
    });

    // Activate initial-v1 and update benchmark baseline accuracy
    const restored = await tx.globalModel.update({
      where: { id: initialV1.id },
      data: {
        isActive: true,
        baselineAccuracy: 95.20,
        baselineLoss: 0.1286,
      },
    });

    // Create audit log entry
    const auditLog = await tx.auditLog.create({
      data: {
        userId: adminUser.id,
        action: 'ROLLBACK_MODEL',
        resourceType: 'GlobalModel',
        resourceId: initialV1.id,
        detailsJson: {
          rolledBackTo: 'initial-v1',
          previousVersion: 'global-v22',
          previousModelId: globalV22.id,
          reason: 'severe validation degradation',
          benchmarkAccuracy: 95.20,
          benchmarkLoss: 0.1286,
          initialV1Sha256: initialV1Hash,
          globalV22PreservedSha256: globalV22Hash,
          timestamp: new Date().toISOString(),
        },
      },
    });

    return { restored, auditLog };
  });

  console.log('Transaction completed successfully.');
  console.log(`Restored Model: ${result.restored.version} (Active: ${result.restored.isActive}, Baseline Acc: ${result.restored.baselineAccuracy}%)`);
  console.log(`Audit Log ID: ${result.auditLog.id}, Action: ${result.auditLog.action}`);

  // 5. Verification
  const activeModel = await prisma.globalModel.findFirst({ where: { isActive: true } });
  console.log(`Verified Active Model in DB: ${activeModel.version} (${activeModel.id})`);
  const v22Check = await prisma.globalModel.findUnique({ where: { id: globalV22.id } });
  console.log(`Verified global-v22 preserved in DB: ${v22Check.version} (Active: ${v22Check.isActive})`);
  console.log(`Verified global-v22 file preserved on disk: ${fs.existsSync(globalV22File)}`);
}

main()
  .catch((err) => {
    console.error('Error during rollback execution:', err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
