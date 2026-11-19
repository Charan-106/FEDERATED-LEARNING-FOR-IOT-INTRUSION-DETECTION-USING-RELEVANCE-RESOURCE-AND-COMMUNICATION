const { PrismaClient } = require('../backend/node_modules/@prisma/client');
const prisma = new PrismaClient();

async function main() {
  const hash = '57b7afd9d01c30b48101fc2660d699ea1b5790104c00e534e00e090b517d4993';
  const updated = await prisma.globalModel.update({
    where: { version: 'initial-v1' },
    data: {
      checksumSha256: hash,
      isActive: true, // Make v1 the active base model for the demo
    }
  });
  // Also deactivate global-v2 if active so initial-v1 is active
  await prisma.globalModel.updateMany({
    where: { version: { not: 'initial-v1' } },
    data: { isActive: false }
  });
  console.log('Updated initial-v1 model:', updated);
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());
