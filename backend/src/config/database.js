// Prisma Client singleton for the ARC-FL Backend
// Instantiated once and reused throughout the application lifecycle

const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient({
  log: process.env.NODE_ENV === 'development'
    ? ['error', 'warn']
    : ['error'],
  errorFormat: 'pretty',
});

// Handle graceful shutdown
process.on('beforeExit', async () => {
  await prisma.$disconnect();
});

module.exports = prisma;
