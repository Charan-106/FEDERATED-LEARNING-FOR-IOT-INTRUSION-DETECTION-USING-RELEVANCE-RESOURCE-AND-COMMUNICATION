const { PrismaClient } = require('../backend/node_modules/@prisma/client');
const prisma = new PrismaClient();

async function main() {
  console.log('[Migration] Updating database schema for Round Management...');
  
  // 1. Add CANCELLED enum value if not exists
  await prisma.$executeRawUnsafe(`
    DO $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_enum e ON t.oid = e.enumtypid WHERE t.typname = 'RoundStatus' AND e.enumlabel = 'CANCELLED') THEN
        ALTER TYPE "RoundStatus" ADD VALUE 'CANCELLED';
      END IF;
    END
    $$;
  `);
  console.log('[Migration] Added CANCELLED to RoundStatus enum.');

  // 2. Add columns to training_rounds table if they do not exist
  await prisma.$executeRawUnsafe(`
    ALTER TABLE "training_rounds"
    ADD COLUMN IF NOT EXISTS "name" TEXT,
    ADD COLUMN IF NOT EXISTS "description" TEXT,
    ADD COLUMN IF NOT EXISTS "target_clients" JSONB,
    ADD COLUMN IF NOT EXISTS "hyperparameters" JSONB,
    ADD COLUMN IF NOT EXISTS "arc_weights" JSONB;
  `);
  console.log('[Migration] Added columns (name, description, target_clients, hyperparameters, arc_weights) to training_rounds.');

  console.log('[Migration] SUCCESS!');
}

main()
  .catch((err) => {
    console.error('[Migration] Failed:', err);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
