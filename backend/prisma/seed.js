// Prisma Seed Script - Creates demo accounts and initial data
// Run with: node prisma/seed.js

require('dotenv').config({ path: '../.env' });

const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcryptjs');

const prisma = new PrismaClient();

const BCRYPT_ROUNDS = 12;

async function main() {
  console.log('[Seed] Starting database seed...');

  // ============================================================
  // 1. Create Admin User
  // ============================================================
  const adminPasswordHash = await bcrypt.hash('Admin@123', BCRYPT_ROUNDS);
  const admin = await prisma.user.upsert({
    where: { email: 'admin@arc-fl.local' },
    update: {},
    create: {
      email: 'admin@arc-fl.local',
      passwordHash: adminPasswordHash,
      name: 'Platform Administrator',
      role: 'ADMIN',
    },
  });
  console.log(`[Seed] Admin created: ${admin.email}`);

  // ============================================================
  // 2. Create Client Users (3 clients matching the 3 datasets)
  // ============================================================
  const clientDefs = [
    { email: 'client1@arc-fl.local', name: 'Edge Node Alpha', clientId: 'client-001' },
    { email: 'client2@arc-fl.local', name: 'Edge Node Beta', clientId: 'client-002' },
    { email: 'client3@arc-fl.local', name: 'Edge Node Gamma', clientId: 'client-003' },
  ];

  const clientPasswordHash = await bcrypt.hash('Client@123', BCRYPT_ROUNDS);

  for (const clientDef of clientDefs) {
    const user = await prisma.user.upsert({
      where: { email: clientDef.email },
      update: {},
      create: {
        email: clientDef.email,
        passwordHash: clientPasswordHash,
        name: clientDef.name,
        role: 'CLIENT',
      },
    });

    await prisma.clientProfile.upsert({
      where: { clientId: clientDef.clientId },
      update: {},
      create: {
        userId: user.id,
        clientId: clientDef.clientId,
        name: clientDef.name,
        status: 'ACTIVE',
        hardwareSpecs: {
          cpu: 'Generic x86-64',
          ram_gb: 8,
          gpu: null,
          os: 'Linux',
        },
      },
    });

    console.log(`[Seed] Client created: ${clientDef.email} (${clientDef.clientId})`);
  }

  // ============================================================
  // 3. Create Initial GlobalModel record (preact_resmlp_v3)
  // ============================================================
  const initialModel = await prisma.globalModel.upsert({
    where: { version: 'initial-v1' },
    update: {},
    create: {
      version: 'initial-v1',
      name: 'CICIoT2023 ResMLP Baseline',
      architecture: 'preact_resmlp_v3',
      inputFeatures: 45,
      numClasses: 34,
      totalParameters: 4981628,
      // storagePath is relative to STORAGE_ROOT
      storagePath: 'global-models/initial-v1',
      isActive: true,
      baselineAccuracy: 98.50,
      baselineLoss: 0.0423,
    },
  });
  console.log(`[Seed] Global model created: ${initialModel.version} (active=${initialModel.isActive})`);

  // ============================================================
  // 4. Create Default ARC Configuration
  // ============================================================
  // Deactivate any existing configs first
  await prisma.aRCConfig.updateMany({ data: { isActive: false } });

  const arcConfig = await prisma.aRCConfig.create({
    data: {
      version: '1.0',
      isActive: true,
      // Top-level lambda weights (sum to 1.0)
      lambdaRelevance: 0.40,      // 40% weight for data relevance
      lambdaResource: 0.30,       // 30% weight for resource capability
      lambdaCommunication: 0.30,  // 30% weight for communication quality
      // Relevance sub-weights (sum to 1.0): R_i = 0.60*Q_i + 0.40*D_i
      weightQuality: 0.60,
      weightDiversity: 0.40,
      // Resource sub-weights (sum to 1.0): S_i = 0.40*M_i + 0.40*P_i + 0.20*A_i
      weightMemory: 0.40,
      weightProcessing: 0.40,
      weightCapacity: 0.20,
      // Communication sub-weights (sum to 1.0): C_i = 0.50*B_i + 0.30*L_i + 0.20*T_i
      weightBandwidth: 0.50,
      weightLatency: 0.30,
      weightReliability: 0.20,
    },
  });
  console.log(`[Seed] ARC config created (v${arcConfig.version})`);

  // ============================================================
  // 5. Summary
  // ============================================================
  console.log('\n[Seed] ✅ Seed complete!');
  console.log('=== Demo Accounts ===');
  console.log('  Admin:    admin@arc-fl.local    / Admin@123');
  console.log('  Client 1: client1@arc-fl.local  / Client@123');
  console.log('  Client 2: client2@arc-fl.local  / Client@123');
  console.log('  Client 3: client3@arc-fl.local  / Client@123');
  console.log('===================');
}

main()
  .catch((err) => {
    console.error('[Seed] Error:', err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
