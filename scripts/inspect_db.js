const prisma = require('../backend/src/config/database');

async function main() {
  const users = await prisma.user.findMany({ select: { id: true, email: true, role: true, name: true } });
  console.log('Users in DB:', JSON.stringify(users, null, 2));

  const clients = await prisma.clientProfile.findMany();
  console.log('Clients in DB:', JSON.stringify(clients, null, 2));

  const rounds = await prisma.trainingRound.findMany();
  console.log('Rounds count in DB:', rounds.length);

  const models = await prisma.globalModel.findMany();
  console.log('Models in DB:', JSON.stringify(models, null, 2));
}

main().catch(console.error).finally(() => prisma.$disconnect());
